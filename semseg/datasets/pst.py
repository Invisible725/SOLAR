# file: pst_dataloader.py
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import torchvision.transforms.functional as TF

from semseg.augmentations_thermal import get_train_augmentation

# --------------------------------------------------------------------- #
#                                PST                                    #
# --------------------------------------------------------------------- #
class PST(Dataset):
    """
    RGB-Thermal semantic-segmentation dataset with given train / test splits.

        root/
            train/
                rgb/*.png
                thermal/*.png
                labels/*.png
            test/
                rgb/*.png
                thermal/*.png
                labels/*.png
    """

    # background is ignored (mapped to 255), the remaining four classes are valid
    CLASSES: List[str] = [
        "background",            # ignore
        "fire extinguisher",
        "backpack",
        "hand drill",
        "rescue randy",
    ]

    PALETTE = torch.tensor(      # adjust if masks use different colours
        [
            (  0,   0,   0),     # background
            (  0,   0, 255),     # fire extinguisher
            (  0, 255,   0),     # backpack
            (255,   0,   0),     # hand drill
            (255, 255,   255),     # rescue randy
        ]
    )

    def __init__(
        self,
        root: str = "data/pst",
        split: str = "train",
        transform=None,
        modals: Tuple[str, ...] = ("rgb", "thermal"),
        subset: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "test"}
        assert set(modals).issubset({"rgb", "thermal"})

        self.root = root
        self.transform = transform
        self.modals = list(modals)

        self.ignore_label = 255
        self.n_classes = len(self.CLASSES) - 1          # four output channels

        pattern = os.path.join(root, split, "rgb", "*.png")
        self.files = sorted(glob.glob(pattern))
        if subset is not None:
            self.files = [p for p in self.files if subset in p]
        if not self.files:
            raise RuntimeError(f"No images found with pattern: {pattern} (subset={subset})")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Tensor, str]:
        rgb_path = Path(self.files[idx])
        folder   = rgb_path.parent.parent

        th_path  = folder / "thermal" / rgb_path.name
        lbl_path = folder / "labels"  / rgb_path.name

        sample: dict = {}

        if "rgb" in self.modals:
            rgb = io.read_image(str(rgb_path))[:3]
            sample["img"] = rgb

        if "thermal" in self.modals:
            th = io.read_image(str(th_path))
            if th.shape[0] == 1:
                th = th.repeat(3, 1, 1)
            sample["thermal"] = th

        sample["mask"] = io.read_image(str(lbl_path))

        if self.transform:
            sample = self.transform(sample)

        y = self._encode(sample["mask"])
        del sample["mask"]

        ordered = [sample["img"] if m == "rgb" else sample[m] for m in self.modals]
        return ordered, y, str(rgb_path)

    # ------------------------------------------------------------------ #
    def _encode(self, lbl: Tensor) -> Tensor:
        """
        background (0) → 255; classes 1–4 → 0–3
        """
        # single-channel mask
        if lbl.ndim == 2 or lbl.shape[0] == 1:
            lbl = lbl.squeeze(0).long()
            out = torch.full_like(lbl, self.ignore_label)
            valid = lbl > 0
            out[valid] = lbl[valid] - 1
            out[out >= self.n_classes] = self.ignore_label
            return out

        # RGB palette mask
        h, w = lbl.shape[1:]
        flat = lbl.permute(1, 2, 0).contiguous().view(-1, 3)
        palette = self.PALETTE.to(flat.device).int()

        out = torch.full((flat.shape[0],), self.ignore_label,
                         dtype=torch.long, device=flat.device)

        for cls_idx in range(1, len(palette)):          # skip background
            color = palette[cls_idx]
            match = torch.all(flat == color, dim=1)
            out[match] = cls_idx - 1

        return out.view(h, w)

# --------------------------------------------------------------------- #
#                             quick test                                #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    root = "/path/to/pst"          # change to actual path
    tf   = get_train_augmentation((640, 512), seg_fill=255)

    tr = PST(root=root, split="train", transform=tf)
    te = PST(root=root, split="test",  transform=tf)

    assert set(tr.files).isdisjoint(te.files)

    print("train images:", len(tr))
    print("test  images:", len(te))

    loader = DataLoader(tr, batch_size=2, num_workers=2)
    for i, (_, y, _) in enumerate(loader):
        print("unique labels:", torch.unique(y))
        if i == 1:
            break
