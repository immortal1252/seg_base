import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms


# # # # # # # # # # # # # # # # # # # # # # # #
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # #


def img_aug_identity(img):
    return img


def img_aug_autocontrast(img):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img):
    return ImageOps.equalize(img)


def img_aug_invert(img):
    return ImageOps.invert(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # # print(f"final:{v}")
    # v = np.random.uniform(scale[0], scale[1])
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=None):
    if scale is None:
        scale = [4, 8]
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=None):
    if scale is None:
        scale = [1, 256]
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)


def get_augment_list(flag_using_wide=False):
    if flag_using_wide:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.1, 1.8]),
            (img_aug_brightness, [0.1, 1.8]),
            (img_aug_color, [0.1, 1.8]),
            (img_aug_sharpness, [0.1, 1.8]),
            (img_aug_posterize, [2, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5]),
        ]
    else:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5]),
        ]
    return l


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1 <= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, scales)
        return img


# cv2的接口
class StrongImageAug:
    def __init__(self, num_augs, flag_using_random_num=False):
        self.strong_aug = strong_img_aug(num_augs, flag_using_random_num)

    def __call__(self, cv2_img):
        # cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img)
        pil_img = self.strong_aug(pil_img)
        cv2_img = np.array(pil_img)
        # cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
        return cv2_img


if __name__ == "__main__":
    aug = strong_img_aug(5)
