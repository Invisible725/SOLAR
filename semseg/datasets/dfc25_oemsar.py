"""DFC2025 Track 1 / OpenEarthMap-SAR dataset loader for the RM_SAM pipeline.

Expected split layout:
  root/
    train/
      rgb_images/
      sar_images/
      labels/
    val/
      rgb_images/
      sar_images/
      labels/
    train.txt
    val.txt

The split manifests can contain either filenames with suffix
(`TrainArea_001.tif`) or bare stems (`TrainArea_001`).
Label values follow the official 8-class IDs 1..8 and are remapped to 0..7.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


_CLASSES_8 = [
    'Bareland',
    'Grass',
    'Pavement',
    'Road',
    'Tree',
    'Water',
    'Cropland',
    'Buildings',
]
_PALETTE_8 = torch.tensor([
    [128, 0, 0],      # Bareland
    [0, 255, 36],     # Grass
    [148, 148, 148],  # Pavement
    [255, 255, 255],  # Road
    [34, 97, 38],     # Tree
    [0, 69, 255],     # Water
    [75, 181, 73],    # Cropland
    [222, 31, 7],     # Buildings
])
_LABEL_MAP = {raw: raw - 1 for raw in range(1, 9)}


class DFC25OEMSAR(Dataset):
    """Pre-split DFC2025 Track 1 dataset with RGB + SAR + label TIFFs."""

    CLASSES = _CLASSES_8
    PALETTE = _PALETTE_8

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        modals=('optical', 'sar'),
        optical_folder: str = 'rgb_images',
        sar_folder: str = 'sar_images',
        label_folder: str = 'labels',
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        split_root = self.root / split
        if not split_root.exists():
            raise FileNotFoundError(f'Split folder not found: {split_root}')

        rgb_dir = split_root / optical_folder
        sar_dir = split_root / sar_folder
        lbl_dir = split_root / label_folder
        for folder in (rgb_dir, sar_dir, lbl_dir):
            if not folder.exists():
                raise FileNotFoundError(f'Required folder not found: {folder}')

        split_file = self.root / f'{split}.txt'
        names = None
        if split_file.exists():
            names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            names = [p.name for p in sorted(lbl_dir.glob('*.tif'))]

        self.samples = []
        for name in names:
            stem = Path(name).stem
            rgb_p = rgb_dir / f'{stem}.tif'
            sar_p = sar_dir / f'{stem}.tif'
            lbl_p = lbl_dir / f'{stem}.tif'
            if not (rgb_p.exists() and sar_p.exists() and lbl_p.exists()):
                missing = [str(p) for p in (rgb_p, sar_p, lbl_p) if not p.exists()]
                raise FileNotFoundError(f'Missing files for sample {name}: {missing}')
            self.samples.append((sar_p, rgb_p, lbl_p))

        print(f'[DFC25OEMSAR] {split}: {len(self.samples)} samples from {split_root}')

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_label(label_path: Path) -> np.ndarray:
        label = np.array(Image.open(label_path).convert('L'), dtype=np.int64)
        remapped = np.full_like(label, 255, dtype=np.int64)
        for raw_val, cls_idx in _LABEL_MAP.items():
            remapped[label == raw_val] = cls_idx
        return remapped

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        sar_p, rgb_p, lbl_p = self.samples[index]

        sar = np.array(Image.open(sar_p).convert('L'))
        rgb = np.array(Image.open(rgb_p).convert('RGB'))
        lbl = self._read_label(lbl_p)

        sar_t = torch.from_numpy(sar).unsqueeze(0)
        rgb_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        lbl_t = torch.from_numpy(lbl.copy()).unsqueeze(0)

        sample = {}
        for modal in self.modals:
            if modal == 'sar':
                sample[modal] = sar_t
            elif modal in ('optical', 'opt', 'rgb'):
                sample[modal] = rgb_t
            else:
                raise ValueError(f'Unsupported modality for DFC25OEMSAR: {modal}')

        sample['mask'] = lbl_t
        sample['img'] = sample[self.modals[0]]

        if self.transform:
            sample = self.transform(sample)

        label_out = sample['mask']
        sample[self.modals[0]] = sample['img']
        del sample['mask'], sample['img']

        return [sample[k] for k in self.modals], label_out.squeeze(0).long()
