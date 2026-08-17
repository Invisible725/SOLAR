"""
WHU-OPT-SAR dataset — aligned with MCANet baseline for fair comparison.

Supports two data layouts:
  A. MCANet pre-tiled: dataset/{train,val,test}/{sar,opt,lbl}/*.tif  (256x256 patches)
  B. Raw large tiles:  whu-opt-sar/{sar,optical,lbl}/*.tif           (tiled on-the-fly)

Key alignment with MCANet:
  - 8 classes (background=0 included), label map: 0→0, 10→1, ..., 70→7
  - Optical: 4 channels (RGBA / RGBNIR), NOT converted to RGB
  - SAR: 1 channel (native grayscale)
"""

import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, List
import glob
from PIL import Image


# ── 8-class setup matching MCANet ──
_CLASSES_8 = ['Background', 'Farmland', 'City', 'Village', 'Water', 'Forest', 'Road', 'Others']
_PALETTE_8 = torch.tensor([
    [0, 0, 0],         # Background
    [0, 102, 204],     # Farmland
    [0, 0, 255],       # City
    [0, 255, 255],     # Village
    [255, 0, 0],       # Water
    [0, 167, 85],      # Forest
    [255, 255, 0],     # Road
    [153, 102, 153],   # Others
])
# Raw label value → class index (MCANet convention)
_LABEL_MAP_8 = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}


class WHUOptSAR(Dataset):
    """
    Pre-tiled dataset (MCANet layout).
    Expects: root/{sar,opt,lbl}/*.tif  — already 256x256 patches.
    Labels already converted to 0-7 by MCANet's convert_label.py.
    """
    CLASSES = _CLASSES_8
    PALETTE = _PALETTE_8

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        modals=('optical', 'sar'),
        **kwargs,  # absorb unused args for compat
    ) -> None:
        super().__init__()
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.root = root

        # MCANet layout after split_data.py: ROOT/{train,val,test}/{sar,opt,lbl}/
        # MCANet layout after crop only:    ROOT/{sars,opticals,lbls}/
        # Our raw layout:                   ROOT/{sar,optical,lbl}/
        if os.path.isdir(os.path.join(root, split)):
            split_root = os.path.join(root, split)
        else:
            split_root = root

        # Find directories — try all naming conventions
        sar_dir = self._find_dir(split_root, ['sar', 'sars'])
        opt_dir = self._find_dir(split_root, ['opt', 'optical', 'opticals'])
        lbl_dir = self._find_dir(split_root, ['lbl', 'label', 'lbls'])

        sar_files = sorted(os.listdir(sar_dir))
        self.samples = []
        for fn in sar_files:
            sar_p = os.path.join(sar_dir, fn)
            opt_p = os.path.join(opt_dir, fn)
            lbl_p = os.path.join(lbl_dir, fn)
            if os.path.exists(opt_p) and os.path.exists(lbl_p):
                self.samples.append((sar_p, opt_p, lbl_p))

        if not self.samples:
            raise FileNotFoundError(f"No matched samples in {split_root}. "
                                    f"sar_dir={sar_dir}, opt_dir={opt_dir}, lbl_dir={lbl_dir}")

        print(f"[WHUOptSAR] {split}: {len(self.samples)} samples from {split_root}")

    @staticmethod
    def _find_dir(root, candidates):
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.isdir(p):
                return p
        existing = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))] if os.path.isdir(root) else []
        raise FileNotFoundError(
            f"None of {candidates} found in {root}. "
            f"Existing subdirs: {existing}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        sar_p, opt_p, lbl_p = self.samples[index]

        # SAR: load as 1-channel grayscale
        sar = np.array(Image.open(sar_p).convert('L'))           # H, W uint8

        # Optical: load as 4-channel RGBA (preserves NIR/alpha band)
        opt_img = Image.open(opt_p)
        if opt_img.mode == 'RGBA':
            opt = np.array(opt_img)                               # H, W, 4
        elif opt_img.mode == 'RGB':
            opt = np.array(opt_img)                               # H, W, 3
        else:
            opt = np.array(opt_img.convert('RGB'))                # H, W, 3

        # Label: already 0-7 if from MCANet's convert_label, or raw 0/10/20/../70
        lbl = np.array(Image.open(lbl_p).convert('L')).astype(np.int64)  # H, W

        # If labels are still in raw format (max > 7), remap
        if lbl.max() > 7:
            remapped = np.zeros_like(lbl)
            for raw_val, cls_idx in _LABEL_MAP_8.items():
                remapped[lbl == raw_val] = cls_idx
            lbl = remapped

        # Build sample dict for augmentation pipeline
        # SAR: 1ch → CHW tensor
        sar_t = torch.from_numpy(sar).unsqueeze(0)                # 1, H, W
        # Optical: HWC → CHW tensor
        opt_t = torch.from_numpy(opt.copy()).permute(2, 0, 1)     # C, H, W
        lbl_t = torch.from_numpy(lbl.copy()).unsqueeze(0)         # 1, H, W

        sample = {}
        for modal in self.modals:
            if modal in ('sar',):
                sample[modal] = sar_t
            elif modal in ('optical', 'opt'):
                sample[modal] = opt_t

        sample['mask'] = lbl_t
        sample['img'] = sample[self.modals[0]]

        if self.transform:
            sample = self.transform(sample)

        label_out = sample['mask']
        sample[self.modals[0]] = sample['img']
        del sample['mask'], sample['img']

        return [sample[k] for k in self.modals], label_out.squeeze(0).long()


class WHUOptSAR_RawTiles(Dataset):
    """
    Raw large-tile dataset with on-the-fly patching + RAM cache.
    For when you have the raw tiles but haven't pre-tiled them.
    """
    CLASSES = _CLASSES_8
    PALETTE = _PALETTE_8

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        modals=('optical', 'sar'),
        patch_size: int = 256,
        stride: int = 256,
        ignore_thresh: float = 0.5,
        split_ratio=(0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.root = root
        self.patch_size = patch_size

        # Find valid tiles
        lbl_dir = os.path.join(root, 'lbl')
        all_lbl = sorted(glob.glob(os.path.join(lbl_dir, '*.tif')))

        # Map modal names to directory names
        modal_dirs = {}
        for m in modals:
            for cand in [m, 'opt' if m == 'optical' else m]:
                d = os.path.join(root, cand)
                if os.path.isdir(d):
                    modal_dirs[m] = cand
                    break

        valid_basenames = []
        for lbl_path in all_lbl:
            bn = os.path.basename(lbl_path)
            if all(os.path.exists(os.path.join(root, modal_dirs[m], bn)) for m in modals):
                valid_basenames.append(bn)

        if not valid_basenames:
            raise FileNotFoundError(f"No valid tiles in {root}")

        # Split at tile level
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(valid_basenames))
        n_train = max(1, int(len(valid_basenames) * split_ratio[0]))
        n_val = max(1, int(len(valid_basenames) * split_ratio[1]))

        if len(valid_basenames) <= 2:
            sel_basenames = valid_basenames
        else:
            if split == 'train':
                sel = indices[:n_train]
            elif split == 'val':
                sel = indices[n_train:n_train + n_val]
            else:
                sel = indices[n_train + n_val:]
            sel_basenames = [valid_basenames[i] for i in sorted(sel)]

        # Cache tiles in RAM
        self._tiles = {}
        for bn in sel_basenames:
            tile = {}
            for m in modals:
                p = os.path.join(root, modal_dirs[m], bn)
                if m == 'sar':
                    tile[m] = np.array(Image.open(p).convert('L'))         # H,W
                else:
                    img = Image.open(p)
                    if img.mode == 'RGBA':
                        tile[m] = np.array(img)                            # H,W,4
                    else:
                        tile[m] = np.array(img.convert('RGB'))             # H,W,3

            lbl_raw = np.array(Image.open(os.path.join(root, 'lbl', bn)).convert('L'))
            lbl = np.zeros_like(lbl_raw, dtype=np.int64)
            for rv, ci in _LABEL_MAP_8.items():
                lbl[lbl_raw == rv] = ci
            tile['lbl'] = lbl
            self._tiles[bn] = tile

        ram_mb = sum(sum(v.nbytes for v in t.values()) for t in self._tiles.values()) / 1e6
        print(f"[WHUOptSAR_RawTiles] Cached {len(sel_basenames)} tiles ({ram_mb:.0f} MB)")

        # Build patch index
        ps = patch_size
        self.patches = []
        for bn, tile in self._tiles.items():
            H, W = tile['lbl'].shape
            for r in range(0, H - ps + 1, stride):
                for c in range(0, W - ps + 1, stride):
                    self.patches.append((bn, r, c))

        if split == 'val' and len(valid_basenames) <= 2:
            rng2 = np.random.RandomState(seed + 1)
            n_val_patches = max(100, len(self.patches) // 5)
            if len(self.patches) > n_val_patches:
                sel_idx = rng2.choice(len(self.patches), n_val_patches, replace=False)
                self.patches = [self.patches[i] for i in sorted(sel_idx)]

        print(f"[WHUOptSAR_RawTiles] {split}: {len(self.patches)} patches ({ps}x{ps}, stride={stride})")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        bn, r, c = self.patches[index]
        ps = self.patch_size
        tile = self._tiles[bn]

        sample = {}
        for modal in self.modals:
            arr = tile[modal]
            if arr.ndim == 2:
                patch = arr[r:r+ps, c:c+ps].copy()
                sample[modal] = torch.from_numpy(patch).unsqueeze(0)      # 1,H,W
            else:
                patch = arr[r:r+ps, c:c+ps, :].copy()
                sample[modal] = torch.from_numpy(patch).permute(2, 0, 1)  # C,H,W

        lbl = tile['lbl'][r:r+ps, c:c+ps].copy()
        sample['mask'] = torch.from_numpy(lbl).unsqueeze(0)
        sample['img'] = sample[self.modals[0]]

        if self.transform:
            sample = self.transform(sample)

        label_out = sample['mask']
        sample[self.modals[0]] = sample['img']
        del sample['mask'], sample['img']

        return [sample[k] for k in self.modals], label_out.squeeze(0).long()
