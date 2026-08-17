"""
Training script for CrossModalSAM2 on WHU-OPT-SAR.
Aligned with MCANet baseline for fair comparison:
  - 8 classes (background included)
  - Reports OA, Kappa, mIoU, per-class F1
  - Supports both pre-tiled and raw-tile datasets
"""
import os
import copy
import torch
import argparse
import yaml
import time
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from PIL import Image

from semseg.datasets.whuoptsar import WHUOptSAR, WHUOptSAR_RawTiles
from semseg.datasets.piergbsar import PIERGBSAR
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, setup_ddp, cleanup_ddp, get_logger, print_iou
from semseg.metrics import Metrics

from sam2.build_sam import build_sam2
from solar import SOLAR, SingleModalSAM2

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'


MODEL_REGISTRY = {
    'SOLAR': SOLAR,
    'SingleModalSAM2': SingleModalSAM2,
}


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema = self._clone_model(model)

    @staticmethod
    def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()
        return ema

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)


def estimate_class_weights(dataset, cfg):
    """Estimate class weights from the training split with conservative scaling.

    The goal is to help minority classes such as Road/Others without letting
    rare classes dominate the optimization and hurt OA.
    """
    loss_cfg = cfg['LOSS']
    mode = loss_cfg.get('WEIGHT_MODE', 'inv_sqrt')
    power = float(loss_cfg.get('WEIGHT_POWER', 0.5))
    max_weight = float(loss_cfg.get('MAX_WEIGHT', 3.0))
    min_weight = float(loss_cfg.get('MIN_WEIGHT', 0.5))

    hist = np.zeros(dataset.n_classes, dtype=np.float64)

    if hasattr(dataset, 'samples'):
        label_paths = [lbl_p for _, _, lbl_p in dataset.samples]
        label_map = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}
        for lbl_p in label_paths:
            if hasattr(dataset, '_read_label'):
                lbl = dataset._read_label(lbl_p)
            else:
                lbl = np.array(Image.open(lbl_p).convert('L'), dtype=np.int64)
                if lbl.max() > dataset.n_classes - 1:
                    remapped = np.zeros_like(lbl)
                    for raw_val, cls_idx in label_map.items():
                        remapped[lbl == raw_val] = cls_idx
                    lbl = remapped
            valid = lbl != dataset.ignore_label
            hist += np.bincount(lbl[valid].reshape(-1), minlength=dataset.n_classes)
    elif hasattr(dataset, '_tiles'):
        for tile in dataset._tiles.values():
            lbl = tile['lbl']
            valid = lbl != dataset.ignore_label
            hist += np.bincount(lbl[valid].reshape(-1), minlength=dataset.n_classes)
    else:
        raise ValueError('Unable to estimate class weights for this dataset layout.')

    hist = np.maximum(hist, 1.0)
    freq = hist / hist.sum()

    if mode == 'inv':
        weights = 1.0 / np.power(freq, power)
    elif mode == 'median':
        weights = np.median(freq) / freq
    else:
        weights = 1.0 / np.sqrt(freq)

    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, dataloader, device, valid_classes=None, original_size=None):
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader, desc='Eval'):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        _, seg_out = model(images)
        if isinstance(seg_out, tuple):
            seg_out = seg_out[0]
        if original_size is not None:
            seg_out = F.interpolate(seg_out, size=original_size, mode='bilinear', align_corners=False)
            labels = F.interpolate(labels.unsqueeze(1).float(), size=original_size, mode='nearest').squeeze(1).long()
        metrics.update(seg_out.softmax(dim=1), labels)

    ious, miou = metrics.compute_iou(valid_classes=valid_classes)
    acc, macc = metrics.compute_pixel_acc(valid_classes=valid_classes)
    f1, mf1 = metrics.compute_f1(valid_classes=valid_classes)
    oa = metrics.compute_oa(valid_classes=valid_classes)
    kappa = metrics.compute_kappa(valid_classes=valid_classes)
    return {
        'oa': oa, 'kappa': kappa,
        'miou': miou, 'macc': macc, 'mf1': mf1,
        'ious': ious, 'acc': acc, 'f1': f1,
    }


def main(cfg, save_dir):
    start = time.time()
    best_OA = 0.0
    best_mIoU = 0.0
    best_Kappa = 0.0
    best_epoch = 0
    device = torch.device(cfg['DEVICE'])

    train_cfg = cfg['TRAIN']
    eval_cfg = cfg['EVAL']
    dataset_cfg = cfg['DATASET']
    loss_cfg = cfg['LOSS']
    optim_cfg = cfg['OPTIMIZER']
    sched_cfg = cfg['SCHEDULER']

    epochs = train_cfg['EPOCHS']
    lr = optim_cfg['LR']
    accum_steps = train_cfg.get('ACCUM_STEPS', 2)
    aux_weight = train_cfg.get('AUX_WEIGHT', 0.4)
    unc_weight = train_cfg.get('UNC_WEIGHT', 0.0)
    modals = dataset_cfg['MODALS']
    eval_classes = dataset_cfg.get('EVAL_CLASSES', None)
    original_size = eval_cfg.get('ORIGINAL_SIZE', None)
    if original_size is not None:
        original_size = tuple(original_size)

    logger.info(f'Train image size: {tuple(train_cfg["IMAGE_SIZE"])}')
    logger.info(f'Eval image size: {tuple(eval_cfg["IMAGE_SIZE"])}')
    logger.info(f'Eval original size: {original_size if original_size is not None else "disabled"}')

    # 鈹€鈹€ Data 鈹€鈹€
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    dataset_name = dataset_cfg['NAME']

    if dataset_name == 'PIERGBSAR':
        optical_folder = dataset_cfg.get('OPTICAL_FOLDER', 'rgb')
        trainset = PIERGBSAR(
            dataset_cfg['ROOT'], 'train', traintransform, modals,
            optical_folder=optical_folder,
        )
        valset = PIERGBSAR(
            dataset_cfg['ROOT'], 'val', valtransform, modals,
            optical_folder=optical_folder,
        )
    elif dataset_name == 'WHUOptSAR':
        data_format = dataset_cfg.get('FORMAT', 'pretiled')  # 'pretiled' or 'raw'
        if data_format == 'pretiled':
            # MCANet-style: dataset/{train,val,test}/{sar,opt,lbl}/
            trainset = WHUOptSAR(dataset_cfg['ROOT'], 'train', traintransform, modals)
            valset = WHUOptSAR(dataset_cfg['ROOT'], 'val', valtransform, modals)
        elif data_format == 'raw':
            # Raw tiles: whu-opt-sar/{sar,optical,lbl}/*.tif
            trainset = WHUOptSAR_RawTiles(
                dataset_cfg['ROOT'], 'train', traintransform, modals,
                patch_size=train_cfg.get('PATCH_SIZE', 256),
                stride=train_cfg.get('STRIDE', 128),
            )
            valset = WHUOptSAR_RawTiles(
                dataset_cfg['ROOT'], 'val', valtransform, modals,
                patch_size=train_cfg.get('PATCH_SIZE', 256),
                stride=train_cfg.get('STRIDE_VAL', 256),
            )
        else:
            raise ValueError(f'Unsupported WHUOptSAR FORMAT: {data_format}. Available: pretiled, raw')
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}. Available: PIERGBSAR, WHUOptSAR')

    class_names = trainset.CLASSES
    class_names_eval = [class_names[i] for i in eval_classes] if eval_classes is not None else class_names

    trainloader = DataLoader(
        trainset, batch_size=train_cfg['BATCH_SIZE'],
        num_workers=8, drop_last=True, pin_memory=True,
        sampler=RandomSampler(trainset),
    )
    valloader = DataLoader(
        valset, batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=8, pin_memory=True,
    )

    # 鈹€鈹€ Model 鈹€鈹€
    sam2 = build_sam2(cfg['MODEL']['SAM2_CFG'], cfg['MODEL']['SAM2_CKPT'])
    model_name = cfg['MODEL'].get('NAME', 'SOLAR')
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model: {model_name}. Available: {list(MODEL_REGISTRY)}')
    model = MODEL_REGISTRY[model_name](
        sam2, modalities=modals,
        num_classes=trainset.n_classes,
        lora_r=train_cfg.get('LORA_R', 16),
        sar_modality=dataset_cfg.get('SAR_MODAL', 'sar'),
        modal_channels=dataset_cfg.get('MODAL_CHANNELS', None),
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable ({100*trainable/total:.1f}%)')
    logger.info(f'Classes ({trainset.n_classes}): {class_names}')

    # 鈹€鈹€ Loss / Optimizer / Scheduler 鈹€鈹€
    cls_weights = None
    zero_weight_classes = dataset_cfg.get('LOSS_ZERO_CLASSES', None)
    if zero_weight_classes is not None:
        cls_weights = torch.ones(trainset.n_classes, dtype=torch.float32)
        for idx in zero_weight_classes:
            cls_weights[idx] = 0.0
    if loss_cfg.get('CLS_WEIGHTS', False):
        cls_weights = estimate_class_weights(trainset, cfg)
        if zero_weight_classes is not None:
            cls_weights[zero_weight_classes] = 0.0
        cls_weights = cls_weights.to(device)
        logger.info(f'Class weights: {[round(float(x), 4) for x in cls_weights.cpu()]}')
    elif cls_weights is not None:
        cls_weights = cls_weights.to(device)
        logger.info(f'Class weights: {[round(float(x), 4) for x in cls_weights.cpu()]}')

    if zero_weight_classes is not None:
        logger.info(
            f'Zero-weight classes: {zero_weight_classes} -> '
            f'{[class_names[i] for i in zero_weight_classes]}'
        )
    if eval_classes is not None:
        logger.info(
            f'Eval classes: {eval_classes} -> '
            f'{class_names_eval}'
        )
    loss_kwargs = {}
    for key, value in loss_cfg.items():
        if key in ('NAME', 'CLS_WEIGHTS', 'WEIGHT_MODE', 'WEIGHT_POWER', 'MIN_WEIGHT', 'MAX_WEIGHT'):
            continue
        loss_kwargs[key.lower()] = value
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, cls_weights, **loss_kwargs)
    iters_per_epoch = len(trainloader)
    total_iters = (epochs + 1) * iters_per_epoch

    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(
        sched_cfg['NAME'], optimizer, total_iters,
        sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'],
    )
    scaler = GradScaler(enabled=train_cfg.get('AMP', False))
    writer = SummaryWriter(str(save_dir))
    ema = None
    ema_enabled = train_cfg.get('EMA', False)
    ema_decay = train_cfg.get('EMA_DECAY', 0.999)
    if ema_enabled:
        ema = ModelEma(model, decay=ema_decay)

    logger.info(f'Train: {len(trainset)}, Val: {len(valset)}, '
                f'BS={train_cfg["BATCH_SIZE"]}, Accum={accum_steps}, LR={lr}, '
                f'Model={model_name}, UNC_WEIGHT={unc_weight}, '
                f'EMA={ema_enabled}, EMA_DECAY={ema_decay if ema_enabled else "off"}')

    # 鈹€鈹€ Training 鈹€鈹€
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f'Epoch [{epoch+1}/{epochs}]')

        for it, (sample, lbl) in pbar:
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)

            with autocast(enabled=train_cfg.get('AMP', False)):
                _, seg_logits, aux = model(sample, return_aux=True)
                loss_main = loss_fn(seg_logits, lbl)
                loss_aux = loss_fn(aux['aux_logits'], lbl)
                loss_unc = aux.get('uncertainty_loss')
                if loss_unc is None:
                    loss_unc = torch.zeros((), device=device, dtype=loss_main.dtype)
                loss = (loss_main + aux_weight * loss_aux + unc_weight * loss_unc) / accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % accum_steps == 0 or (it + 1) == iters_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            train_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f'{train_loss/(it+1):.4f}')

        train_loss /= iters_per_epoch
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        # 鈹€鈹€ Eval 鈹€鈹€
        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and
                (epoch + 1) > train_cfg.get('EVAL_START', 0)) or (epoch + 1) == epochs:

            eval_model = ema.ema if ema is not None else model
            res = evaluate(
                eval_model, valloader, device,
                valid_classes=eval_classes,
                original_size=original_size,
            )
            writer.add_scalar('val/OA', res['oa'], epoch)
            writer.add_scalar('val/Kappa', res['kappa'], epoch)
            writer.add_scalar('val/mIoU', res['miou'], epoch)

            logger.info(
                f"Epoch {epoch+1}: OA={res['oa']:.2f}%  Kappa={res['kappa']:.2f}%  "
                f"mIoU={res['miou']:.2f}%  mF1={res['mf1']:.2f}%  "
                f"(best OA={best_OA:.2f}%, best mIoU={best_mIoU:.2f}%, best Kappa={best_Kappa:.2f}%)"
            )

            is_better = (
                (res['oa'] > best_OA) or
                (res['oa'] == best_OA and res['miou'] > best_mIoU) or
                (res['oa'] == best_OA and res['miou'] == best_mIoU and res['kappa'] > best_Kappa)
            )

            if is_better:
                old = save_dir / f'best_ep{best_epoch}_{best_OA:.2f}.pth'
                if old.exists():
                    old.unlink()
                best_OA = res['oa']
                best_mIoU = res['miou']
                best_Kappa = res['kappa']
                best_epoch = epoch + 1
                ckpt = save_dir / f'best_ep{best_epoch}_{best_OA:.2f}.pth'
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_oa': best_OA,
                    'best_miou': best_mIoU,
                    'best_kappa': best_Kappa,
                    'ema_enabled': ema is not None,
                }, ckpt)
                logger.info(print_iou(epoch, res['ious'], res['miou'], res['acc'], res['macc'], class_names_eval))

    writer.close()
    end = time.gmtime(time.time() - start)

    # 鈹€鈹€ Final summary 鈹€鈹€
    logger.info('\n' + '='*60)
    logger.info('FINAL RESULTS (best epoch):')
    logger.info(tabulate([
        ['Best OA', f"{best_OA:.2f}%"],
        ['Best mIoU', f"{best_mIoU:.2f}%"],
        ['Best Kappa', f"{best_Kappa:.2f}%"],
        ['Best Epoch', best_epoch],
        ['Time', time.strftime('%H:%M:%S', end)],
    ], numalign='right'))

    # Re-evaluate best model for full report
    best_ckpt = save_dir / f'best_ep{best_epoch}_{best_OA:.2f}.pth'
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model_state_dict'])
        res = evaluate(
            model, valloader, device,
            valid_classes=eval_classes,
            original_size=original_size,
        )
        logger.info(f"\nBest model metrics:")
        logger.info(f"  OA    = {res['oa']:.2f}%")
        logger.info(f"  Kappa = {res['kappa']:.2f}%")
        logger.info(f"  mIoU  = {res['miou']:.2f}%")
        logger.info(f"  mF1   = {res['mf1']:.2f}%")
        logger.info(f"  Per-class IoU: {dict(zip(class_names_eval, res['ious']))}")
        logger.info(f"  Per-class F1:  {dict(zip(class_names_eval, res['f1']))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/whuoptsar_crossmodal.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    setup_ddp()

    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    exp_name = f"{cfg['DATASET']['NAME']}_{cfg['MODEL']['BACKBONE']}_{modals}"
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')

    main(cfg, save_dir)
    cleanup_ddp()
