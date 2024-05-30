import cv2
import pandas
from torch.utils.data import DataLoader
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import albumentations as A


class BreastDataset(Dataset):
    def __init__(self, path="", phase="train", transforms=None):
        self.phase = phase
        self.path = path
        if phase == "train":
            self.filename_list = os.listdir(os.path.join(path, "img_crop50"))
        else:
            self.filename_list = os.listdir(os.path.join(path, "img_crop50"))
        self.transforms = transforms

    def __getitem__(self, index):
        if self.phase == "train":
            image_name = self.filename_list[index]  # 19-2016_214.png
            # print(image_name)
            image_path = os.path.join(self.path, "img_crop50", image_name)
            mask_path = os.path.join(self.path, "gt_crop50", image_name)
        else:
            image_name = self.filename_list[index]
            image_path = os.path.join(self.path, "img_crop50", image_name)
            mask_path = os.path.join(self.path, "gt_crop50", image_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image.transpose((2, 0, 1))
        image = image / 255
        mask = mask / 255

        if self.transforms:
            augmented = self.transforms()(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return (
            torch.as_tensor(image.copy()).float().contiguous(),
            torch.as_tensor(mask.copy()).float().contiguous().unsqueeze(0),
        )

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":

    def _get_train_transformers():
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.6, rotate_limit=45, p=1.0
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
                    ],
                    p=1.0,
                ),
            ],
            p=0.9,
        )

    dataset_path = "../data/train"
    train_loader = DataLoader(
        dataset=BreastDataset(
            path=dataset_path, phase="train", transforms=_get_train_transformers
        ),
        batch_size=2,
        num_workers=2,
        drop_last=True,
        shuffle=True,
    )

    # 2730
    for i, ct in enumerate(train_loader):
        print(i, ct["image"].size(), ct["mask"].size())
