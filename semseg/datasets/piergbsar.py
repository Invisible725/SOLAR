"""PIE-RGB-SAR dataset loader for the RM_SAM training pipeline.

Expected layout:
  PIE-RGB-SAR/
    rgb/
    rgb_clouds/
    sar/
    label/
    train.txt
    val.txt

`train.txt` / `val.txt` contain sample ids without suffix.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


_CLASSES_6 = ['Other', 'City', 'Road', 'Water', 'Forest', 'Farmland']
_PALETTE_6 = torch.tensor([
    [128, 128, 128],  # Other / background-like class
    [235, 51, 36],    # City
    [255, 253, 85],   # Road
    [66, 236, 245],   # Water
    [56, 168, 0],     # Forest
    [168, 112, 0],    # Farmland
])


class PIERGBSAR(Dataset):
    """Pre-tiled PIE-RGB-SAR dataset using official train.txt / val.txt splits."""

    CLASSES = _CLASSES_6
    PALETTE = _PALETTE_6

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        modals=('optical', 'sar'),
        optical_folder: str = 'rgb',
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals
        self.optical_folder = optical_folder
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        split_file = self.root / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')

        rgb_dir = self.root / optical_folder
        sar_dir = self.root / 'sar'
        lbl_dir = self.root / 'label'
        for folder in [rgb_dir, sar_dir, lbl_dir]:
            if not folder.exists():
                raise FileNotFoundError(f'Required folder not found: {folder}')

        names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.samples = []
        for name in names:
            rgb_p = rgb_dir / f'{name}.tif'
            sar_p = sar_dir / f'{name}.tif'
            lbl_p = lbl_dir / f'{name}.tif'
            if not (rgb_p.exists() and sar_p.exists() and lbl_p.exists()):
                missing = [str(p) for p in [rgb_p, sar_p, lbl_p] if not p.exists()]
                raise FileNotFoundError(f'Missing files for sample {name}: {missing}')
            self.samples.append((sar_p, rgb_p, lbl_p))

        print(f"[PIERGBSAR] {split}: {len(self.samples)} samples from {self.root} (optical_folder={optical_folder})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        sar_p, rgb_p, lbl_p = self.samples[index]

        sar = np.array(Image.open(sar_p).convert('L'))                 # H, W
        rgb = np.array(Image.open(rgb_p).convert('RGB'))               # H, W, 3
        lbl = np.array(Image.open(lbl_p).convert('L')).astype(np.int64)

        sar_t = torch.from_numpy(sar).unsqueeze(0)                     # 1, H, W
        rgb_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1)          # 3, H, W
        lbl_t = torch.from_numpy(lbl.copy()).unsqueeze(0)              # 1, H, W

        sample = {}
        for modal in self.modals:
            if modal == 'sar':
                sample[modal] = sar_t
            elif modal in ('optical', 'opt', 'rgb'):
                sample[modal] = rgb_t
            else:
                raise ValueError(f'Unsupported modality for PIERGBSAR: {modal}')

        sample['mask'] = lbl_t
        sample['img'] = sample[self.modals[0]]

        if self.transform:
            sample = self.transform(sample)

        label_out = sample['mask']
        sample[self.modals[0]] = sample['img']
        del sample['mask'], sample['img']

        return [sample[k] for k in self.modals], label_out.squeeze(0).long()
