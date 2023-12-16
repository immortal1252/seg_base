import albumentations as A
import torch
import cv2
from .augs_TIBA import StrongImageAug
from albumentations.pytorch import ToTensorV2

# BUSI
MEAN, STD = 0.32963505387306213, 0.2224094420671463
# dataset_b
# MEAN, STD = 0.24798217310678738, 0.15146150910781206
norm = A.Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))
resize = A.Resize(448, 448)
totensor = ToTensorV2()
norm_totensor = A.Compose([norm, totensor])
basic_aug = A.Compose(
    [
        A.RandomResizedCrop(448, 448, scale=(0.5, 1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Resize(448, 448),
    ]
)

strong_aug = StrongImageAug(7, True)

from torchvision.transforms.transforms import ToPILImage
import numpy as np


def debug(img, fp="new.png"):
    if isinstance(img, np.ndarray):
        img = totensor(image=img)["image"]
    img = img.cpu()
    ToPILImage()(img).save(fp)


def mask2tensor(mask):
    assert mask.ndim == 2
    mask = torch.from_numpy(mask) / 255.0
    mask = mask.unsqueeze(0)
    return mask


def cut_mix(tensor1, tensor2, mask):
    return tensor1 * mask + (1 - mask) * tensor2
