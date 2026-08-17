import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam2.build_sam import build_sam2
from semseg.augmentations_mm import get_val_augmentation
from semseg.datasets.piergbsar import PIERGBSAR
from semseg.datasets.whuoptsar import WHUOptSAR, WHUOptSAR_RawTiles
from semseg.metrics import Metrics
from semseg.utils.utils import fix_seeds, setup_cudnn
from solar import SOLAR, SingleModalSAM2


MODEL_REGISTRY = {
    'SOLAR': SOLAR,
    'SingleModalSAM2': SingleModalSAM2,
}


def build_dataset(cfg, split):
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    train_cfg = cfg.get('TRAIN', {})
    modals = dataset_cfg['MODALS']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset_name = dataset_cfg['NAME']

    if dataset_name == 'PIERGBSAR':
        return PIERGBSAR(
            dataset_cfg['ROOT'],
            split,
            transform,
            modals,
            optical_folder=dataset_cfg.get('OPTICAL_FOLDER', 'rgb'),
        )

    if dataset_name == 'WHUOptSAR':
        data_format = dataset_cfg.get('FORMAT', 'pretiled')
        if data_format == 'pretiled':
            return WHUOptSAR(dataset_cfg['ROOT'], split, transform, modals)
        if data_format == 'raw':
            return WHUOptSAR_RawTiles(
                dataset_cfg['ROOT'],
                split,
                transform,
                modals,
                patch_size=train_cfg.get('PATCH_SIZE', 256),
                stride=train_cfg.get('STRIDE_VAL', 256),
            )
        raise ValueError(f'Unsupported WHUOptSAR FORMAT: {data_format}. Available: pretiled, raw')

    raise ValueError(f'Unsupported dataset: {dataset_name}. Available: PIERGBSAR, WHUOptSAR')


def build_model(cfg, num_classes, device):
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    train_cfg = cfg.get('TRAIN', {})
    model_name = model_cfg.get('NAME', 'SOLAR')
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model: {model_name}. Available: {list(MODEL_REGISTRY)}')

    sam2 = build_sam2(model_cfg['SAM2_CFG'], model_cfg['SAM2_CKPT'])
    model = MODEL_REGISTRY[model_name](
        sam2,
        modalities=dataset_cfg['MODALS'],
        num_classes=num_classes,
        lora_r=train_cfg.get('LORA_R', 16),
        sar_modality=dataset_cfg.get('SAR_MODAL', 'sar'),
        modal_channels=dataset_cfg.get('MODAL_CHANNELS', None),
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    return msg


@torch.no_grad()
def evaluate(model, dataloader, device, valid_classes=None, original_size=None):
    model.eval()
    dataset = dataloader.dataset
    metrics = Metrics(dataset.n_classes, dataset.ignore_label, device)

    for images, labels in tqdm(dataloader, desc='Test'):
        images = [x.to(device, non_blocking=True) for x in images]
        labels = labels.to(device, non_blocking=True)

        _, logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        if original_size is not None:
            logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
            labels = F.interpolate(labels.unsqueeze(1).float(), size=original_size, mode='nearest').squeeze(1).long()

        metrics.update(logits.softmax(dim=1), labels)

    ious, miou = metrics.compute_iou(valid_classes=valid_classes)
    acc, macc = metrics.compute_pixel_acc(valid_classes=valid_classes)
    f1, mf1 = metrics.compute_f1(valid_classes=valid_classes)
    oa = metrics.compute_oa(valid_classes=valid_classes)
    kappa = metrics.compute_kappa(valid_classes=valid_classes)
    return {
        'oa': oa,
        'kappa': kappa,
        'miou': miou,
        'macc': macc,
        'mf1': mf1,
        'ious': ious,
        'acc': acc,
        'f1': f1,
    }


def resolve_checkpoint(cfg, model_path):
    if model_path:
        checkpoint_path = Path(model_path)
    else:
        if 'MODEL_PATH' not in cfg['EVAL']:
            raise ValueError('Please set EVAL.MODEL_PATH in the config or pass --model-path.')
        checkpoint_path = Path(cfg['EVAL']['MODEL_PATH'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Test SOLAR or SingleModalSAM2 checkpoints.')
    parser.add_argument('--cfg', type=str, required=True, help='Path to train/test yaml config.')
    parser.add_argument('--model-path', type=str, default=None, help='Checkpoint path. Overrides EVAL.MODEL_PATH.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    with open(args.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(args.seed)
    setup_cudnn()

    device = torch.device(cfg.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = build_dataset(cfg, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    checkpoint_path = resolve_checkpoint(cfg, args.model_path)
    model = build_model(cfg, dataset.n_classes, device)
    load_msg = load_checkpoint(model, checkpoint_path, device)

    eval_classes = cfg['DATASET'].get('EVAL_CLASSES', None)
    class_names = dataset.CLASSES
    class_names_eval = [class_names[i] for i in eval_classes] if eval_classes is not None else class_names
    original_size = cfg['EVAL'].get('ORIGINAL_SIZE', None)
    if original_size is not None:
        original_size = tuple(original_size)

    print(f'Testing {cfg["MODEL"].get("NAME", "SOLAR")} on {cfg["DATASET"]["NAME"]}/{args.split}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Load state: {load_msg}')
    start = time.time()
    res = evaluate(model, dataloader, device, valid_classes=eval_classes, original_size=original_size)

    table = {
        'Class': list(class_names_eval) + ['Mean'],
        'IoU': res['ious'] + [res['miou']],
        'F1': res['f1'] + [res['mf1']],
        'Acc': res['acc'] + [res['macc']],
    }
    print(tabulate(table, headers='keys', floatfmt='.2f'))
    print(
        f"OA={res['oa']:.2f}%  Kappa={res['kappa']:.2f}%  "
        f"mIoU={res['miou']:.2f}%  mF1={res['mf1']:.2f}%  "
        f"Time={time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}"
    )


if __name__ == '__main__':
    main()
