import torch
from torch import Tensor
from typing import Optional, Sequence, Tuple


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes, device=device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes ** 2,
        ).view(self.num_classes, self.num_classes)

    def _select_hist(self, valid_classes: Optional[Sequence[int]] = None) -> Tensor:
        if valid_classes is None:
            return self.hist.float()
        idx = torch.as_tensor(valid_classes, device=self.hist.device, dtype=torch.long)
        return self.hist.index_select(0, idx).index_select(1, idx).float()

    def compute_iou(
        self, valid_classes: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, Tensor]:
        hist = self._select_hist(valid_classes)
        ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
        ious[ious.isnan()] = 0.0
        miou = ious.mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(
        self, valid_classes: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, Tensor]:
        hist = self._select_hist(valid_classes)
        f1 = 2 * hist.diag() / (hist.sum(0) + hist.sum(1))
        f1[f1.isnan()] = 0.0
        mf1 = f1.mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(
        self, valid_classes: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, Tensor]:
        hist = self._select_hist(valid_classes)
        acc = hist.diag() / hist.sum(1)
        acc[acc.isnan()] = 0.0
        macc = acc.mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def compute_oa(self, valid_classes: Optional[Sequence[int]] = None) -> float:
        hist = self._select_hist(valid_classes)
        correct = hist.diag().sum().item()
        total = hist.sum().item()
        return round(100.0 * correct / max(total, 1), 2)

    def compute_kappa(self, valid_classes: Optional[Sequence[int]] = None) -> float:
        hist = self._select_hist(valid_classes)
        total = hist.sum().item()
        if total == 0:
            return 0.0
        po = hist.diag().sum().item() / total
        row_sums = hist.sum(dim=1)
        col_sums = hist.sum(dim=0)
        pe = (row_sums * col_sums).sum().item() / (total * total)
        if pe >= 1.0:
            return 0.0
        kappa = (po - pe) / (1.0 - pe)
        return round(100.0 * kappa, 2)
