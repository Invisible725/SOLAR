"""
CrossModalSAM2 — Multi-modal semantic segmentation with SAM2 backbone.

Architecture (novel):
  ┌──────────────────────────────────────────────────┐
  │ Optical → InputAdapter → ┐                        │
  │                           ├→ Shared SAM2+LoRA ×2  │
  │ SAR     → InputAdapter → ┘    (same weights!)     │
  │                                                    │
  │ Features at 3 scales: 1/4, 1/8, 1/16             │
  │                                                    │
  │ Cross-modal attention at 1/16 bottleneck           │
  │ Spatial-gated fusion at 1/8 and 1/4               │
  │                                                    │
  │ Multi-scale SegHead → logits + aux logits          │
  └──────────────────────────────────────────────────┘

Key novelty: single shared SAM2 (NOT duplicated per modality).
- Modality input adapters bridge domain gap (especially for SAR)
- Shared LoRA learns cross-modal feature space
- Cross-attention enables feature exchange
- ~50% fewer params than per-modality duplication
"""

import copy
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sam2.modeling.sam2_base import SAM2Base


# ──────────────────────────────────────────────
# Utility modules
# ──────────────────────────────────────────────

def _freeze_except_norms(module: nn.Module):
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                  nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)
    for m in module.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad = False
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                p.requires_grad = True


class _LoRALayer(nn.Module):
    """Standard LoRA on a linear layer."""
    def __init__(self, original: nn.Linear, r: int, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        in_dim, out_dim = original.in_features, original.out_features
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.original(x) + self.scale * self.B(self.drop(self.A(x)))


def _inject_lora(model: SAM2Base, r: int, alpha: float = 16.0, dropout: float = 0.05):
    """Inject LoRA into QKV projections of Hiera trunk blocks."""
    trunk = model.image_encoder.trunk
    for blk in trunk.blocks:
        old_qkv = blk.attn.qkv
        blk.attn.qkv = _LoRALayer(old_qkv, r, alpha, dropout)


# ──────────────────────────────────────────────
# Input adapters
# ──────────────────────────────────────────────

class InputAdapter(nn.Module):
    """Projects any-channel modality input to 3ch for SAM2.
    SAR (1ch) or Optical (4ch) → 3ch RGB-like representation.
    When in_ch==3, uses gated residual (identity + learned delta).
    When in_ch!=3, learns a full projection."""
    def __init__(self, in_ch=3, mid_ch=32, out_ch=3):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
        )
        self.skip = None
        if in_ch == out_ch:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            # Learnable skip: project in_ch → out_ch with 1x1 conv
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        projected = self.net(x)
        if self.skip is not None:
            return self.skip(x) + self.gate.tanh() * projected
        return x + self.gate.tanh() * projected


# ──────────────────────────────────────────────
# Cross-modal attention (bottleneck only)
# ──────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention with FFN and gated residual."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.xattn_a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.xattn_b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_a = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim*2, dim))
        self.ffn_b = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim*2, dim))
        self.gate_a = nn.Parameter(torch.zeros(1))
        self.gate_b = nn.Parameter(torch.zeros(1))

    def forward(self, fa_4d, fb_4d):
        B, C, H, W = fa_4d.shape
        fa = fa_4d.flatten(2).transpose(1, 2)
        fb = fb_4d.flatten(2).transpose(1, 2)
        xa, _ = self.xattn_a2b(self.norm_a(fa), self.norm_b(fb), fb)
        xb, _ = self.xattn_b2a(self.norm_b(fb), self.norm_a(fa), fa)
        xa = xa + self.ffn_a(xa)
        xb = xb + self.ffn_b(xb)
        oa = fa_4d + self.gate_a.tanh() * xa.transpose(1, 2).view(B, C, H, W)
        ob = fb_4d + self.gate_b.tanh() * xb.transpose(1, 2).view(B, C, H, W)
        return oa, ob


# ──────────────────────────────────────────────
# Spatial-gated fusion
# ──────────────────────────────────────────────

class SpatialGate(nn.Module):
    """Context-aware 7x7 gated fusion for 2 feature maps."""
    def __init__(self, ch, ks=7):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, ks, padding=ks//2, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        alpha = self.gate(torch.cat([a, b], 1))
        return alpha * a + (1 - alpha) * b


class UncertaintyAwareFuse(nn.Module):
    """Fuse two feature maps with learned aleatoric uncertainty.

    Each modality predicts a per-pixel log-variance. Precision-weighted fusion
    suppresses unreliable regions, while a lightweight spatial gate preserves
    complementary detail from both branches.
    """
    def __init__(self, ch, hidden_ch=None, logvar_min=-4.0, logvar_max=4.0):
        super().__init__()
        hidden_ch = hidden_ch or max(ch // 2, 16)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.var_a = nn.Sequential(
            nn.Conv2d(ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        self.var_b = nn.Sequential(
            nn.Conv2d(ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        self.refine = SpatialGate(ch)

    def forward(self, a, b):
        logvar_a = self.var_a(a).clamp_(self.logvar_min, self.logvar_max)
        logvar_b = self.var_b(b).clamp_(self.logvar_min, self.logvar_max)
        prec_a = torch.exp(-logvar_a)
        prec_b = torch.exp(-logvar_b)

        fused_unc = (prec_a * a + prec_b * b) / (prec_a + prec_b + 1e-6)
        fused_gate = self.refine(a, b)
        confidence = torch.sigmoid(0.5 * (prec_a + prec_b))
        delta = (fused_unc - fused_gate) * confidence
        fused = fused_gate + self.res_scale.tanh() * delta

        unc_loss = (
            (prec_a * (a - fused_gate.detach()).pow(2) + 0.1 * logvar_a).mean() +
            (prec_b * (b - fused_gate.detach()).pow(2) + 0.1 * logvar_b).mean()
        ) * 0.5

        stats = {
            'uncertainty_loss': unc_loss,
            'logvar_a': logvar_a,
            'logvar_b': logvar_b,
            'confidence': confidence,
        }
        return fused, stats


# ──────────────────────────────────────────────
# Segmentation head with deep supervision
# ──────────────────────────────────────────────

class SegHead(nn.Module):
    """Multi-scale fusion segmentation head.
    Returns (main_logits, aux_logits) during training.
    main_logits at full input resolution, aux at 1/8 scale upsampled."""
    def __init__(self, c16, c8, c4, num_classes):
        super().__init__()
        # 1/16 → compress
        self.aspp = nn.Sequential(
            nn.Conv2d(c16, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=6, dilation=6, groups=128, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.proj8 = nn.Sequential(nn.Conv2d(c8, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.proj4 = nn.Sequential(nn.Conv2d(c4, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))

        self.fuse = nn.Sequential(
            nn.Conv2d(128+64+64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(128, num_classes, 1)

        # Aux head at 1/8
        self.aux = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, f16, f8, f4):
        x = self.aspp(f16)
        x8 = F.interpolate(x, size=f8.shape[2:], mode='bilinear', align_corners=False)
        p8 = self.proj8(f8)

        # Aux at 1/8
        aux = self.aux(torch.cat([x8, p8], 1))

        # Main at 1/4
        x4 = F.interpolate(x8, size=f4.shape[2:], mode='bilinear', align_corners=False)
        p8_4 = F.interpolate(p8, size=f4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.proj4(f4)
        out = self.fuse(torch.cat([x4, p8_4, p4], 1))
        out = self.cls(self.drop(out))

        # Upsample to full res (f4 is 1/4)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        aux = F.interpolate(aux, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out, aux


class DetailRefineHead(nn.Module):
    """Low-level detail refinement head.

    Uses shallow fused features plus cross-modal disagreement cues to predict a
    small residual on top of the main segmentation logits. The residual gate is
    initialized at zero so training starts from the baseline model behavior.
    """
    def __init__(self, c8, c4, num_classes):
        super().__init__()
        self.proj_f8 = nn.Sequential(
            nn.Conv2d(c8, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.proj_f4 = nn.Sequential(
            nn.Conv2d(c4, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.proj_d8 = nn.Sequential(
            nn.Conv2d(c8, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.proj_d4 = nn.Sequential(
            nn.Conv2d(c4, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, num_classes, 1),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, fused4, fused8, diff4, diff8):
        f8 = F.interpolate(self.proj_f8(fused8), size=fused4.shape[2:], mode='bilinear', align_corners=False)
        d8 = F.interpolate(self.proj_d8(diff8), size=fused4.shape[2:], mode='bilinear', align_corners=False)
        f4 = self.proj_f4(fused4)
        d4 = self.proj_d4(diff4)
        residual = self.refine(torch.cat([f4, f8, d4, d8], dim=1))
        residual = F.interpolate(residual, scale_factor=4, mode='bilinear', align_corners=False)
        return self.gate.tanh() * residual


# ──────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────

class CrossModalSAM2(nn.Module):
    """
    Multi-modal SAM2 for semantic segmentation.

    Novel design:
      - SINGLE shared SAM2 backbone (not duplicated per modality)
      - Per-modality input adapters (bridge domain gap, especially SAR)
      - Shared LoRA in QKV — learns cross-modal feature space
      - Cross-attention at 1/16 for inter-modal feature exchange
      - Spatial-gated fusion at all 3 FPN scales
      - Multi-scale seg head with deep supervision

    ~50% fewer params than per-modality SAM2 duplication.
    """
    def __init__(
        self,
        sam_model: SAM2Base,
        modalities: list,
        num_classes: int = 8,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        sar_modality: str = 'sar',
        modal_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.num_classes = num_classes
        self.sar_modality = sar_modality

        # --- Single shared SAM2 backbone ---
        self.sam = sam_model
        _freeze_except_norms(self.sam)
        _inject_lora(self.sam, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # Unfreeze FPN neck (lightweight, helps adaptation)
        for p in self.sam.image_encoder.neck.parameters():
            p.requires_grad = True

        # --- Per-modality input adapters ---
        # SAR: 1ch → 3ch,  Optical: 4ch → 3ch (or 3ch → 3ch)
        self.adapters = nn.ModuleDict()
        modal_channels = dict(modal_channels or {})
        modal_channels.setdefault(sar_modality, 1)
        for m in modalities:
            in_ch = modal_channels.get(m, 4)  # keep WHU default as 4ch optical
            self.adapters[m] = InputAdapter(in_ch, 32, 3)

        # Channel dims (after forward_image with use_high_res_features_in_sam)
        c16 = sam_model.sam_mask_decoder.transformer_dim   # 256
        c4 = c16 // 8                                     # 32
        c8 = c16 // 4                                     # 64

        # --- Cross-modal attention at bottleneck ---
        self.cross_attn = CrossModalAttention(c16, num_heads=8)

        # --- Spatial-gated fusion at each scale ---
        self.fuse16 = SpatialGate(c16)
        self.fuse8  = SpatialGate(c8)
        self.fuse4  = SpatialGate(c4)

        # --- Segmentation head ---
        self.seg_head = SegHead(c16, c8, c4, num_classes)

        # For compat with training script
        self.top_k = self.num_modalities

    def _encode(self, x, modality):
        """Encode one modality through adapter + shared SAM2."""
        x = self.adapters[modality](x)
        out = self.sam.forward_image(x)
        return {
            'f4':  out['backbone_fpn'][0],    # 32ch
            'f8':  out['backbone_fpn'][1],    # 64ch
            'f16': out['vision_features'],    # 256ch
        }

    def forward(self, batched_input, multimask_output=True, return_aux=False, **kw):
        assert len(batched_input) == self.num_modalities

        # Encode each modality through the SAME backbone
        feats = {}
        for modal, x in zip(self.modalities, batched_input):
            feats[modal] = self._encode(x, modal)

        m0, m1 = self.modalities[0], self.modalities[1]

        # Cross-modal attention at 1/16
        f16_0, f16_1 = self.cross_attn(feats[m0]['f16'], feats[m1]['f16'])

        # Spatial-gated fusion
        fused16 = self.fuse16(f16_0, f16_1)
        fused8  = self.fuse8(feats[m0]['f8'], feats[m1]['f8'])
        fused4  = self.fuse4(feats[m0]['f4'], feats[m1]['f4'])

        # Seg head
        logits, aux_logits = self.seg_head(fused16, fused8, fused4)

        if return_aux:
            return None, logits, {'aux_logits': aux_logits}
        return None, logits


class SingleModalSAM2(nn.Module):
    """Single-modality SAM2 semantic segmentation baseline.

    Uses the same adapter, shared SAM2+LoRA setup, and segmentation head as the
    multimodal model, but removes all cross-modal fusion modules for a clean
    single-modal ablation.
    """
    def __init__(
        self,
        sam_model: SAM2Base,
        modalities: list,
        num_classes: int = 8,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        sar_modality: str = 'sar',
        modal_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if len(modalities) != 1:
            raise ValueError(f'SingleModalSAM2 expects one modality, got {modalities}')
        self.modalities = modalities
        self.num_modalities = 1
        self.num_classes = num_classes

        self.sam = sam_model
        _freeze_except_norms(self.sam)
        _inject_lora(self.sam, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        for p in self.sam.image_encoder.neck.parameters():
            p.requires_grad = True

        modal = modalities[0]
        modal_channels = dict(modal_channels or {})
        modal_channels.setdefault(sar_modality, 1)
        in_ch = modal_channels.get(modal, 4)
        self.adapter = InputAdapter(in_ch, 32, 3)

        c16 = sam_model.sam_mask_decoder.transformer_dim
        c4 = c16 // 8
        c8 = c16 // 4
        self.seg_head = SegHead(c16, c8, c4, num_classes)
        self.top_k = 1

    def _encode(self, x):
        x = self.adapter(x)
        out = self.sam.forward_image(x)
        return {
            'f4': out['backbone_fpn'][0],
            'f8': out['backbone_fpn'][1],
            'f16': out['vision_features'],
        }

    def forward(self, batched_input, multimask_output=True, return_aux=False, **kw):
        assert len(batched_input) == 1
        feats = self._encode(batched_input[0])
        logits, aux_logits = self.seg_head(feats['f16'], feats['f8'], feats['f4'])

        if return_aux:
            return None, logits, {'aux_logits': aux_logits}
        return None, logits


class UncertaintyFusionSAM2(CrossModalSAM2):
    """CrossModalSAM2 with uncertainty-aware multi-scale fusion."""
    def __init__(
        self,
        sam_model: SAM2Base,
        modalities: list,
        num_classes: int = 8,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        sar_modality: str = 'sar',
        modal_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__(
            sam_model,
            modalities=modalities,
            num_classes=num_classes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sar_modality=sar_modality,
            modal_channels=modal_channels,
        )

        c16 = sam_model.sam_mask_decoder.transformer_dim
        c4 = c16 // 8
        c8 = c16 // 4
        self.fuse16 = UncertaintyAwareFuse(c16)
        self.fuse8 = UncertaintyAwareFuse(c8)
        self.fuse4 = UncertaintyAwareFuse(c4)

    def forward(self, batched_input, multimask_output=True, return_aux=False, **kw):
        assert len(batched_input) == self.num_modalities

        feats = {}
        for modal, x in zip(self.modalities, batched_input):
            feats[modal] = self._encode(x, modal)

        m0, m1 = self.modalities[0], self.modalities[1]
        f16_0, f16_1 = self.cross_attn(feats[m0]['f16'], feats[m1]['f16'])

        fused16, unc16 = self.fuse16(f16_0, f16_1)
        fused8, unc8 = self.fuse8(feats[m0]['f8'], feats[m1]['f8'])
        fused4, unc4 = self.fuse4(feats[m0]['f4'], feats[m1]['f4'])

        logits, aux_logits = self.seg_head(fused16, fused8, fused4)
        uncertainty_loss = (
            unc16['uncertainty_loss'] +
            0.2 * unc8['uncertainty_loss'] +
            0.05 * unc4['uncertainty_loss']
        )

        if return_aux:
            return None, logits, {
                'aux_logits': aux_logits,
                'uncertainty_loss': uncertainty_loss,
                'uncertainty_maps': {
                    'f16': (unc16['logvar_a'], unc16['logvar_b']),
                    'f8': (unc8['logvar_a'], unc8['logvar_b']),
                    'f4': (unc4['logvar_a'], unc4['logvar_b']),
                },
            }
        return None, logits


class SOLAR(CrossModalSAM2):
    """SOLAR segmentation model with a conservative detail residual branch."""
    def __init__(
        self,
        sam_model: SAM2Base,
        modalities: list,
        num_classes: int = 8,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        sar_modality: str = 'sar',
        modal_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__(
            sam_model,
            modalities=modalities,
            num_classes=num_classes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sar_modality=sar_modality,
            modal_channels=modal_channels,
        )
        c16 = sam_model.sam_mask_decoder.transformer_dim
        c4 = c16 // 8
        c8 = c16 // 4
        self.detail_head = DetailRefineHead(c8, c4, num_classes)

    def forward(self, batched_input, multimask_output=True, return_aux=False, **kw):
        assert len(batched_input) == self.num_modalities

        feats = {}
        for modal, x in zip(self.modalities, batched_input):
            feats[modal] = self._encode(x, modal)

        m0, m1 = self.modalities[0], self.modalities[1]
        f16_0, f16_1 = self.cross_attn(feats[m0]['f16'], feats[m1]['f16'])

        fused16 = self.fuse16(f16_0, f16_1)
        fused8 = self.fuse8(feats[m0]['f8'], feats[m1]['f8'])
        fused4 = self.fuse4(feats[m0]['f4'], feats[m1]['f4'])

        logits, aux_logits = self.seg_head(fused16, fused8, fused4)

        diff8 = torch.abs(feats[m0]['f8'] - feats[m1]['f8'])
        diff4 = torch.abs(feats[m0]['f4'] - feats[m1]['f4'])
        logits = logits + self.detail_head(fused4, fused8, diff4, diff8)

        if return_aux:
            return None, logits, {'aux_logits': aux_logits}
        return None, logits
