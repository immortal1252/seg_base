import albumentations as A

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
        A.RandomResizedCrop(448, 448, scale=(0.5, 1)),
        A.HorizontalFlip(p=0.5),
    ]
)

strong_aug = StrongImageAug(7, True)
