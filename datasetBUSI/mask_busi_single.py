from datasetBUSI.mask_busi import MaskBUSI
import aug.aug_albu


class MaskBUSISingle(MaskBUSI):
    def __init__(self, root_dir, image_names_path, mode):
        super().__init__(root_dir, image_names_path, mode)

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        image = image.mean(0, keepdim=True)
        return image, mask
