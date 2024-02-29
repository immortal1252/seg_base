from .base_busi import BaseBUSI
import aug.aug_albu


class MaskBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names_path, mode):
        super().__init__(root_dir, image_names_path, mode, get_mask=True)

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        image = aug.aug_albu.norm_totensor(image=augmented["image"])["image"]
        mask = aug.aug_albu.mask2tensor(augmented["mask"])

        return image, mask
