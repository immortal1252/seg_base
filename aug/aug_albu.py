import albumentations as A

import cv2
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2

# # BUSI
MEAN, STD = 0.32963505387306213, 0.2224094420671463
# # dataset_b
# # MEAN, STD = 0.24798217310678738, 0.15146150910781206
norm = A.Normalize((MEAN, MEAN, MEAN), (STD, STD, STD))
resize = A.Resize(448, 448)
totensor = ToTensorV2()
norm_totensor = A.Compose([norm, totensor])


def mask2tensor(mask):
    assert mask.ndim == 2
    mask = torch.from_numpy(mask) / 255.0
    mask = mask.unsqueeze(0)
    return mask


basic_aug = A.Compose(
    [
        A.RandomResizedCrop(448, 448, scale=(0.5, 1)),
        A.HorizontalFlip(p=0.5),
    ]
)
#
P = 0.8
strong_aug = A.Compose(
    [
        A.Equalize(p=P),
        A.GaussianBlur(p=P),
        A.RandomBrightnessContrast(p=P),
        A.HueSaturationValue(p=P),
        A.Posterize(p=P),
        A.Solarize(p=P),
    ]
)
if __name__ == "__main__":
    img = Image.open("../normal (1).png")
    img_cv = cv2.imread("../normal (1).png")
    aug = strong_aug
    img1 = aug(image=img_cv)
    cv2.imshow("cv2", img1["image"])
    cv2.waitKey(0)
    # img2 = augs_TIBA.img_aug_solarize(img)
    # img2.show()
    # cv2.waitKey()
