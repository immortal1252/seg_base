from typing import Dict
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join, splitext
from torchvision.transforms import transforms
import re
import albumentations as A
import cv2
import torch.utils.data
import einops
import aug.aug_albu


def get_mean_std(tensor: torch.Tensor):
    """
    tensor(b,c,h,w) --> list[c], list[c]
    """
    cbhw = einops.rearrange(tensor, "b c h w -> c (b h w)")
    mean = cbhw.mean(1)
    std = cbhw.std(1)
    return mean.tolist(), std.tolist()


def visual(tensor: torch.Tensor):
    """
    show torch.Tensor(1,c,h,w)
    """
    array = tensor.squeeze() * 255
    # if array.ndim == 3:
    if array.dim() == 3:
        array = array.permute(1, 2, 0)
    array = array.cpu().detach().numpy().astype("uint8")
    img = Image.fromarray(array)
    img.show()


def save_img_from_tensor(tensor: torch.Tensor, filename):
    """
    save torch.Tensor(c,h,w)
    """
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor2img = transforms.ToPILImage()
    img = tensor2img(tensor.cpu().float())
    img.save(filename)


def get_with_basic_aug(img_fp, mask_fp, mode, get_mask) -> Dict:
    image = cv2.imread(img_fp)
    mask = None
    if get_mask:
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)

    # 监督训练
    if mode == "train" and mask is not None:
        augmented = aug.aug_albu.basic_aug(image=image, mask=mask)
    # 监督测试
    elif mask is not None:
        augmented = aug.aug_albu.resize(image=image, mask=mask)
    # 无监督训练
    else:
        augmented = aug.aug_albu.basic_aug(image=image)

    return augmented


# 基类,只进行弱数据增强,返回cv2图像
class BaseBUSI(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_names_path, mode, get_mask):
        super().__init__()
        image_names_path = join(root_dir, image_names_path)
        with open(image_names_path, mode="r") as f:
            self.image_names = f.read().splitlines()
        self.mode = mode
        self.root_dir = root_dir
        self.get_mask = get_mask

    def __getitem__(self, index):
        img_name = self.image_names[index]
        for label in ["benign", "malignant"]:
            if label in img_name:
                break
        else:
            raise Exception(f"wrong label:{img_name}")

        mask_name, ext = splitext(img_name)
        mask_name = f"{mask_name}_mask{ext}"
        img_fp = join(self.root_dir, label, "image", img_name)
        mask_fp = join(self.root_dir, label, "mask", mask_name)

        return get_with_basic_aug(img_fp, mask_fp, self.mode, self.get_mask)

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    pass
