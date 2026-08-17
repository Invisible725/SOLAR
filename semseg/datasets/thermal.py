# file: thermal_singles_dataloader.py
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import torchvision.transforms.functional as TF

from semseg.augmentations_thermal import get_train_augmentation   # same helper used by DELIVER

# ------------------------------------------------------------------------- #
#                       split behaviour configuration                       #
# ------------------------------------------------------------------------- #
_SPLIT_RATIO = 0.8   # 80 % train, 20 % test
_SPLIT_SEED  = 42    # keeps the partition fixed across runs


class ThermalSingles(Dataset):
    """
    RGB-Thermal semantic-segmentation dataset with an internal train / test split.

    Directory layout (date / flight vary):
        root/2021-09-09-KentuckyRiver/flight1-1/color/*.png
        root/2021-09-09-KentuckyRiver/flight1-1/thermal8/*.png
        root/2021-09-09-KentuckyRiver/flight1-1/masks_vis/*.png
    """

    CLASSES: List[str] = [
        "unknown",                 # ignored → 255
        "background",
        "bare ground",
        "boulders / rocky terrain",
        "human-made structures",
        "road",
        "shrubs",
        "trees",
        "sky",
        "water",
        "vehicles",
        "person",
    ]

    PALETTE = torch.tensor(
        [
            (255,  36,   0),  # unknown
            (  0,   0,   0),  # background
            (242, 216, 196),  # bare ground
            ( 89,  70,  54),  # boulders
            (166, 166, 166),  # structures
            ( 82,  89,  90),  # road
            (155, 230,   0),  # shrubs
            (  0, 138,  53),  # trees
            (  0, 216, 245),  # sky
            ( 13, 127, 252),  # water
            (255, 249,   0),  # vehicles
            (254,   0, 170),  # person
        ]
    )

    def __init__(
        self,
        root: str = "data/labeled_thermal_singles",
        split: str = "train",           # now really used
        transform=None,
        modals: Tuple[str, ...] = ("rgb", "thermal"),
        subset: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        root      : dataset root
        split     : "train" or "test" (80 % / 20 % fixed split)
        transform : callable that acts on a dict
        modals    : ("rgb", "thermal"), choose any order / subset
        subset    : optional string; only keep samples whose path contains it
        """
        super().__init__()
        assert split in {"train", "test"}, "`split` must be 'train' or 'test'"
        assert set(modals).issubset({"rgb", "thermal"})
        self.root = root
        self.transform = transform
        self.modals = list(modals)
        self.ignore_label = 255
        self.n_classes = len(self.CLASSES) - 2  # unknown ignored

        pattern = os.path.join(root, "*", "*", "color", "*.png")
        all_files = sorted(glob.glob(pattern))
        if subset is not None:
            all_files = [p for p in all_files if subset in p]

        if not all_files:
            raise RuntimeError(f"No images found with pattern: {pattern} (subset={subset})")

        # ----------------------------------------------------------------- #
        #            deterministic in-memory split (train / test)            #
        # ----------------------------------------------------------------- #
        g = torch.Generator().manual_seed(_SPLIT_SEED)
        perm = torch.randperm(len(all_files), generator=g).tolist()
        cut = int(len(perm) * _SPLIT_RATIO)
        train_idx, test_idx = perm[:cut], perm[cut:]

        chosen_idx = train_idx if split == "train" else test_idx
        self.files = [all_files[i] for i in chosen_idx]

        if not self.files:
            raise RuntimeError(
                f"No images found for split '{split}'. "
                f"Check _SPLIT_RATIO or provide more data."
            )

    # ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Tensor]:
        rgb_path = Path(self.files[idx])
        folder = rgb_path.parent.parent     # .../flight*/ 

        th_path = folder / "thermal8" / rgb_path.name
        msk_path = folder / "masks_vis" / rgb_path.name

        sample = {}
        target_size = (640, 512)  # H, W to match thermal and mask

        if "rgb" in self.modals:
            rgb = io.read_image(str(rgb_path))[:3]                      # force 3-channel
            rgb = TF.resize(rgb, target_size, interpolation=TF.InterpolationMode.BILINEAR)
            sample["img"] = rgb

        if "thermal" in self.modals:
            th = io.read_image(str(th_path))
            if th.shape[0] == 1:
                th = th.repeat(3, 1, 1)
            th = TF.resize(th, target_size, interpolation=TF.InterpolationMode.BILINEAR)
            sample["thermal"] = th

        mask = io.read_image(str(msk_path))                              # shape: [3, H, W]
        mask = TF.resize(mask, target_size, interpolation=TF.InterpolationMode.NEAREST)
        sample["mask"] = mask

        if self.transform:
            sample = self.transform(sample)

        y = self._encode(sample["mask"])
        # y[y == 255] = 0                                                  # unknown / background → ignore
        del sample["mask"]

        ordered = [sample["img"] if m == "rgb" else sample[m] for m in self.modals]
        return ordered, y , str(rgb_path)

    # ------------------------------------------------------------ #
    def _encode(self, lbl: Tensor) -> Tensor:
        if lbl.ndim == 2:
            raise ValueError("Expected RGB mask, got single channel")

        h, w = lbl.shape[1:]
        lbl = lbl.permute(1, 2, 0).contiguous().view(-1, 3)              # (H × W, 3)

        palette = self.PALETTE.to(lbl.device).int()
        out = torch.full((lbl.shape[0],), self.ignore_label,
                         dtype=torch.long, device=lbl.device)

        # skip first two: unknown and background → ignore
        for new_class_idx, palette_idx in enumerate(range(2, len(palette))):
            color = palette[palette_idx]
            match = torch.all(lbl == color, dim=1)
            out[match] = new_class_idx                                   # map to 0–9

        return out.view(h, w)


# ------------------------------------------------------------------------- #
#                              sanity check                                 #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    root = "/home/ubuntu/zcy/data/labeled_thermal_singles"

    train_tf = get_train_augmentation((1024, 1024), seg_fill=255)

    # verify both splits are non-empty and mutually exclusive
    train_set = ThermalSingles(root=root, split="train",
                               transform=train_tf, modals=("rgb", "thermal"))
    test_set  = ThermalSingles(root=root, split="test",
                               transform=train_tf, modals=("rgb", "thermal"))

    assert set(train_set.files).isdisjoint(test_set.files), "Train / test overlap!"

    print(f"Train images: {len(train_set):6d}")
    print(f"Test  images: {len(test_set):6d}")

    loader = DataLoader(train_set, batch_size=2, num_workers=2,
                        drop_last=False, pin_memory=False)
    for i, (sample, lbl) in enumerate(loader):
        print("[DEBUG] unique labels:", torch.unique(lbl))
        if i == 1:                                                       # two batches are enough
            break
