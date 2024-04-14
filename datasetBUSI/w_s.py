import numpy as np
import torch
from torchvision.transforms.transforms import ToPILImage
from datasetBUSI.base_busi import BaseBUSI
import aug.aug_albu
from PIL import Image


def visual(img):
    if isinstance(img, torch.Tensor):
        img = img.transpose(0, 2).detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
    pass


class WeakStrongBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names_path, mode, s_num=1):
        # super().__init__(root_dir, image_names_path, mode, get_mask=False)
        super().__init__(root_dir, image_names_path, mode, get_mask=True)
        self.s_num = s_num

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        weak, mask = augmented["image"], augmented["mask"]
        mask = aug.aug_albu.mask2tensor(mask)

        strong_list = []
        for _ in range(self.s_num):
            strong = aug.aug_albu.strong_aug(image=weak)["image"]
            strong = aug.aug_albu.norm_totensor(image=strong)["image"]
            strong_list.append(strong)

        weak = aug.aug_albu.norm_totensor(image=weak)["image"]
        # visual(strong2)
        # mask 不看就好了
        return weak, mask, *strong_list
