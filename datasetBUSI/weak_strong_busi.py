from base_busi import BaseBUSI
import aug.aug_albu


class WeakStrongBUSI(BaseBUSI):
    def __init__(self, root_dir, image_names_path, mode):
        # super().__init__(root_dir, image_names, mode, get_mask=False)
        super().__init__(root_dir, image_names_path, mode, get_mask=True)

    def __getitem__(self, index):
        augmented = super().__getitem__(index)
        weak, mask = augmented["image"], augmented["mask"]
        mask = aug.aug_albu.mask2tensor(mask)
        strong = aug.aug_albu.strong_aug(weak)

        weak = aug.aug_albu.norm_totensor(image=weak)["image"]
        strong = aug.aug_albu.norm_totensor(image=strong)["image"]
        # mask 不看就好了
        return weak, mask, strong
