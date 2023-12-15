from .base_busi import BaseBUSI
from .aug import aug


class MaskBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names, mode):
        super().__init__(root_dir, image_names, mode, get_mask=True)

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        image = aug.norm_totensor(image=augmented["image"])["image"]
        mask = aug.mask2tensor(augmented["mask"])

        return image, mask
