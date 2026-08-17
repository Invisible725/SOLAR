import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        img_keys = ['frame_camera', 'lidar', 'radar', 'event_camera']
        mask = sample['mask']
        img_shapes = [sample[key].shape for key in img_keys if key in sample]
        for img_shape in img_shapes:
            if mask.ndim == 2:
                assert img_shape[1:] == mask.shape
            else:
                assert img_shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Normalize:
    def __init__(self, mean: dict = None, std: dict = None):
        if mean is None:
            mean = {
                'frame_camera': [0.485, 0.456, 0.406],
                'lidar': [0.3850018, 0.47924727, 0.00796207],
                'radar': [0.2816276, 0.46062768, 0.0],
                'event_camera': [0.03909789, 0.03835985, 0.0]
            }
        if std is None:
            std = {
                'frame_camera': [0.229, 0.224, 0.225],
                'lidar': [3.92596071, 5.09148818, 0.3687036],
                'radar': [3.65370654, 4.53079779,  1e-6],
                'event_camera': [0.29815072, 0.27124918,  1e-6]
            }
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':
                continue
            sample[k] = v.float()
            sample[k] /= 255
            if k in self.mean:
                sample[k] = TF.normalize(sample[k], self.mean[k], self.std[k])
        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['frame_camera'] = TF.adjust_brightness(sample['frame_camera'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['frame_camera'] = TF.adjust_contrast(sample['frame_camera'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['frame_camera'] = TF.adjust_saturation(sample['frame_camera'], self.saturation)
        return sample


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sample['frame_camera'] = TF.adjust_sharpness(sample['frame_camera'], self.sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sample['frame_camera'] = TF.autocontrast(sample['frame_camera'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sample['frame_camera'] = TF.gaussian_blur(sample['frame_camera'], self.kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
        return sample


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
        return sample


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        self.size = size

    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, self.size, TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, self.size, TF.InterpolationMode.BILINEAR)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        H, W = sample['frame_camera'].shape[1:]
        tH, tW = self.size

        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        scale = int(tH * ratio), int(tW * ratio)
        scale_factor = min(max(scale) / max(H, W), min(scale) / min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        margin_h = max(sample['frame_camera'].shape[1] - tH, 0)
        margin_w = max(sample['frame_camera'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h + 1)
        x1 = random.randint(0, margin_w + 1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        if sample['frame_camera'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['frame_camera'].shape[2], tH - sample['frame_camera'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample


def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomColorJitter(p=0.2),
        RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur(kernel_size=3, p=0.2),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        Normalize(mean={
            'frame_camera': [0.485, 0.456, 0.406],
            'lidar': [0.3850018, 0.47924727, 0.00796207],
            'radar': [0.2816276, 0.46062768, 0.0],
            'event_camera': [0.03909789, 0.03835985, 0.0]
        }, std={
            'frame_camera': [0.229, 0.224, 0.225],
            'lidar': [3.92596071, 5.09148818, 0.3687036],
            'radar': [3.65370654, 4.53079779,  1e-6],
            'event_camera': [0.29815072, 0.27124918,  1e-6]
        })
    ])

def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        Resize(size),
        Normalize(mean={
            'frame_camera': [0.485, 0.456, 0.406],
            'lidar': [0.3850018, 0.47924727, 0.00796207],
            'radar': [0.2816276, 0.46062768, 0.0],
            'event_camera': [0.03909789, 0.03835985, 0.0]
        }, std={
            'frame_camera': [0.229, 0.224, 0.225],
            'lidar': [3.92596071, 5.09148818, 0.3687036],
            'radar': [3.65370654, 4.53079779,  1e-6],
            'event_camera': [0.29815072, 0.27124918,  1e-6]
        })
    ])