import albumentations as A

import cv2
from .augs_TIBA import StrongImageAug
from albumentations.pytorch import ToTensorV2
import torch

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
        A.RandomResizedCrop(448, 448, scale=(0.5, 1)),
        A.HorizontalFlip(p=0.5),
    ]
)

strong_aug = StrongImageAug(11, True)

from torchvision.transforms.transforms import ToTensor, ToPILImage
import numpy as np


def debug(img, fp="new.png"):
    if isinstance(img, np.ndarray):
        img = ToTensor()(img)
    img = img.cpu()
    ToPILImage()(img).save(fp)


def random_mask(b, h, w):
    # tensor1和2大小一样
    mask_batch = torch.zeros(b, 1, h, w)
    # 随机生成矩形
    size = torch.randint(int(h / 4), int(h / 3), (b,))
    ratio = (1.33 - 0.75) * torch.rand((b,)) + 0.75

    ratio_h = torch.sqrt(ratio)
    ratio_w = 1 / ratio_h
    height = ratio_h * size
    width = ratio_w * size

    for i in range(b):
        # 随机生成左上角位置
        assert w - width[i] > 0
        assert h - height[i] > 0
        rand_x = np.random.randint(0, w - width[i])
        rand_y = np.random.randint(0, h - height[i])
        mask_batch[
            i, 0, rand_y : rand_y + int(height[i]), rand_x : rand_x + int(width[i])
        ] = 1

    return mask_batch
