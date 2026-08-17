"""YESeg-OPT-SAR dataset loader for the RM_SAM training pipeline.

Expected layout:
  YESeg-OPT-SAR/
    optical/
    sar/
    label/
    train.txt
    val.txt

`train.txt` / `val.txt` contain relative file paths such as `1/10.png`.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


_CLASSES_8 = [
    'Background',
    'Bare Ground',
    'Low Vegetation',
    'Trees',
    'Houses',
    'Water',
    'Roads',
    'Other',
]
_PALETTE_8 = torch.tensor([
    [0, 0, 0],
    [128, 64, 0],
    [0, 255, 0],
    [34, 139, 34],
    [255, 0, 0],
    [0, 0, 255],
    [255, 255, 0],
    [128, 128, 128],
])
_FALLBACK_GRAY_TO_INDEX = {
    0: 0,
    66: 1,
    81: 2,
    88: 3,
    116: 4,
    142: 5,
    178: 6,
    255: 7,
}


class YESegOptSAR(Dataset):
    """Pre-tiled YESeg-OPT-SAR dataset using train.txt / val.txt splits."""

    CLASSES = _CLASSES_8
    PALETTE = _PALETTE_8

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        modals=('optical', 'sar'),
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        split_file = self.root / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')

        optical_dir = self.root / 'optical'
        sar_dir = self.root / 'sar'
        label_dir = self.root / 'label'
        for folder in [optical_dir, sar_dir, label_dir]:
            if not folder.exists():
                raise FileNotFoundError(f'Required folder not found: {folder}')

        names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.samples = []
        for name in names:
            rel = Path(name)
            optical_p = optical_dir / rel
            sar_p = sar_dir / rel
            label_p = label_dir / rel
            if not (optical_p.exists() and sar_p.exists() and label_p.exists()):
                missing = [str(p) for p in [optical_p, sar_p, label_p] if not p.exists()]
                raise FileNotFoundError(f'Missing files for sample {name}: {missing}')
            self.samples.append((sar_p, optical_p, label_p))

        print(f"[YESegOptSAR] {split}: {len(self.samples)} samples from {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_label(label_path: Path) -> np.ndarray:
        with Image.open(label_path) as img:
            # Palette PNGs should be read without convert('L'); the raw array
            # then contains class indices 0..7 directly.
            if img.mode == 'P':
                label = np.array(img, dtype=np.int64)
            else:
                label = np.array(img, dtype=np.int64)
                if label.ndim == 3:
                    label = label[..., 0]

        if label.max() > 7:
            remapped = np.zeros_like(label)
            for raw_val, cls_idx in _FALLBACK_GRAY_TO_INDEX.items():
                remapped[label == raw_val] = cls_idx
            label = remapped

        return label

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        sar_p, optical_p, label_p = self.samples[index]

        sar = np.array(Image.open(sar_p).convert('L'))
        optical = np.array(Image.open(optical_p).convert('RGB'))
        label = self._read_label(label_p)

        sar_t = torch.from_numpy(sar).unsqueeze(0)
        optical_t = torch.from_numpy(optical.copy()).permute(2, 0, 1)
        label_t = torch.from_numpy(label.copy()).unsqueeze(0)

        sample = {}
        for modal in self.modals:
            if modal == 'sar':
                sample[modal] = sar_t
            elif modal in ('optical', 'opt', 'rgb'):
                sample[modal] = optical_t
            else:
                raise ValueError(f'Unsupported modality for YESegOptSAR: {modal}')

        sample['mask'] = label_t
        sample['img'] = sample[self.modals[0]]

        if self.transform:
            sample = self.transform(sample)

        label_out = sample['mask']
        sample[self.modals[0]] = sample['img']
        del sample['mask'], sample['img']

        return [sample[k] for k in self.modals], label_out.squeeze(0).long()
