import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class DiceCE(nn.Module):
    """Combined Dice + CrossEntropy loss for better handling of class imbalance."""
    def __init__(self, ignore_label: int = 255, weight: Tensor = None,
                 dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        self.ignore_label = ignore_label

    def _dice(self, preds: Tensor, labels: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        # mask out ignore pixels
        valid = (labels != self.ignore_label)
        labels_valid = labels.clone()
        labels_valid[~valid] = 0
        one_hot = F.one_hot(labels_valid, num_classes).permute(0, 3, 1, 2).float()
        # zero out ignored positions
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = F.softmax(preds, dim=1)
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        ce_loss = self.ce(preds, labels)
        dice_loss = self._dice(preds, labels)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class DiceOhem(nn.Module):
    """Dice + OHEM cross-entropy.

    This keeps the global overlap term from Dice while letting CE focus only on
    the hardest pixels, which is often a better compromise than focal when OA
    is already strong and we mainly want to fix hard boundaries / tail regions.
    """
    def __init__(
        self,
        ignore_label: int = 255,
        weight: Tensor = None,
        dice_weight: float = 1.0,
        ohem_weight: float = 1.0,
        thresh: float = 0.7,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ohem_weight = ohem_weight
        self.ignore_label = ignore_label
        self.ohem = OhemCrossEntropy(ignore_label, weight, thresh=thresh)

    def _dice(self, preds: Tensor, labels: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        valid = (labels != self.ignore_label)
        labels_valid = labels.clone()
        labels_valid[~valid] = 0
        one_hot = F.one_hot(labels_valid, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = F.softmax(preds, dim=1)
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        ohem_loss = self.ohem(preds, labels)
        dice_loss = self._dice(preds, labels)
        return self.ohem_weight * ohem_loss + self.dice_weight * dice_loss


class DiceCeOhem(nn.Module):
    """Dice + CE + OHEM.

    A compromise between global calibration (CE) and hard-pixel mining (OHEM).
    This is useful when OHEM improves OA/Kappa but starts to bias predictions
    toward dominant classes and hurts mean IoU on smaller classes.
    """
    def __init__(
        self,
        ignore_label: int = 255,
        weight: Tensor = None,
        dice_weight: float = 1.0,
        ce_weight: float = 0.5,
        ohem_weight: float = 0.5,
        thresh: float = 0.65,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ohem_weight = ohem_weight
        self.ignore_label = ignore_label
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        self.ohem = OhemCrossEntropy(ignore_label, weight, thresh=thresh)

    def _dice(self, preds: Tensor, labels: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        valid = (labels != self.ignore_label)
        labels_valid = labels.clone()
        labels_valid[~valid] = 0
        one_hot = F.one_hot(labels_valid, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = F.softmax(preds, dim=1)
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        dice_loss = self._dice(preds, labels)
        ce_loss = self.ce(preds, labels)
        ohem_loss = self.ohem(preds, labels)
        return (
            self.dice_weight * dice_loss +
            self.ce_weight * ce_loss +
            self.ohem_weight * ohem_loss
        )


class Focal(nn.Module):
    def __init__(
        self,
        ignore_label: int = 255,
        weight: Tensor = None,
        gamma: float = 1.5,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        ce = F.cross_entropy(
            preds,
            labels,
            weight=self.weight,
            ignore_index=self.ignore_label,
            reduction='none',
        )
        valid = labels != self.ignore_label
        ce = ce[valid]
        pt = torch.exp(-ce)
        focal = self.alpha * ((1 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        return self._forward(preds, labels)


class DiceFocal(nn.Module):
    """Dice + focal CE, suitable when hard pixels matter more than raw frequency."""
    def __init__(
        self,
        ignore_label: int = 255,
        weight: Tensor = None,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 1.5,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = Focal(ignore_label, weight, gamma=gamma, alpha=alpha)
        self.ignore_label = ignore_label

    def _dice(self, preds: Tensor, labels: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        valid = (labels != self.ignore_label)
        labels_valid = labels.clone()
        labels_valid[~valid] = 0
        one_hot = F.one_hot(labels_valid, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = F.softmax(preds, dim=1)
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        focal_loss = self.focal(preds, labels)
        dice_loss = self._dice(preds, labels)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class DiceCEFocal(nn.Module):
    """Dice + CE + focal.

    CE keeps dominant classes stable, while focal still pushes hard pixels.
    This is more conservative than pure DiceFocal and is a better fit when
    OA is already strong and we only want a mild gain on tail classes.
    """
    def __init__(
        self,
        ignore_label: int = 255,
        weight: Tensor = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.5,
        gamma: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        self.focal = Focal(ignore_label, weight, gamma=gamma, alpha=alpha)
        self.ignore_label = ignore_label

    def _dice(self, preds: Tensor, labels: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        valid = (labels != self.ignore_label)
        labels_valid = labels.clone()
        labels_valid[~valid] = 0
        one_hot = F.one_hot(labels_valid, num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = F.softmax(preds, dim=1)
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return (1 - dice).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            preds = preds[0]
        ce_loss = self.ce(preds, labels)
        focal_loss = self.focal(preds, labels)
        dice_loss = self._dice(preds, labels)
        return (
            self.ce_weight * ce_loss +
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss
        )


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'DiceCE', 'DiceOhem', 'DiceCeOhem', 'Focal', 'DiceFocal', 'DiceCEFocal']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None, **kwargs):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights, **kwargs)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)
