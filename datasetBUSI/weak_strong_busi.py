from .base_busi import BaseBUSI
from .aug import aug


class WeakStrongBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names, mode):
        # super().__init__(root_dir, image_names, mode, get_mask=False)
        super().__init__(root_dir, image_names, mode, get_mask=True)

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        weak, mask = augmented["image"], augmented["mask"]
        mask = aug.mask2tensor(mask)
        # strong = aug.strong_aug(weak)

        weak = aug.norm_totensor(image=weak)["image"]
        # strong = aug.norm_totensor(image=strong)["image"]
        # mask 不看就好了
        return weak, mask
