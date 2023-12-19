from .base_busi import BaseBUSI
from .aug import aug
import random
import torch


class MaskBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names, mode):
        super().__init__(root_dir, image_names, mode, get_mask=True)

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        image = aug.norm_totensor(image=augmented["image"])["image"]
        mask = aug.mask2tensor(augmented["mask"])

        return image, mask

    def get_random(self, batch_size):
        idx_list = random.sample(list(range(len(self))), batch_size)
        image_batch = []
        mask_batch = []
        for idx in idx_list:
            image, mask = self[idx]
            image_batch.append(image)
            mask_batch.append(mask)

        image_batch = torch.stack(image_batch)
        mask_batch = torch.stack(mask_batch)

        return image_batch, mask_batch
