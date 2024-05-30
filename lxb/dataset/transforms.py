"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize
import scipy.ndimage as ndimage


from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
# ----------------------data augment-------------------------------------------

# ------------------- 我加 ------------------
# 随机缩放 尺寸没变
class Random_zoom:
    def __init__(self, min_percentage=0.7, max_percentage=1.2, prob=0.5):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage
        self.prob = prob

    def __call__(self, img, msk):
        img = img.numpy()
        msk = msk.numpy()
        img = img.squeeze(0)
        msk = msk.squeeze(0)
        prob = (random.uniform(0, 1))
        if prob < self.prob:
            z = np.random.sample() * (self.max_percentage - self.min_percentage) + self.min_percentage
            zoom_matrix = np.array([[z, 0, 0, 0],
                                    [0, z, 0, 0],
                                    [0, 0, z, 0],
                                    [0, 0, 0, 1]])

            img = ndimage.interpolation.affine_transform(img, zoom_matrix)
            msk = ndimage.interpolation.affine_transform(msk, zoom_matrix)
        img = torch.FloatTensor(img).unsqueeze(0)  # 在第一维增加一个维度
        msk = msk.astype(float)
        msk = torch.FloatTensor(msk).unsqueeze(0)
        #print(img.shape, msk.shape)
        return img, msk


# 弹性变换
class ElasticTransform3d:
    def __init__(self, alpha=1, sigma=20, bg_val=0.0, prob=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.bg_val = bg_val
        self.prob = prob
    def _elastic_transform(self,image, labels,prob=None,alpha=1, sigma=20, bg_val=0.0):
        if prob < self.prob:
            assert image.ndim == 3
            shape = image.shape
            dtype = image.dtype

            # Define coordinate system
            coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

            # Initialize interpolators
            im_intrps = RegularGridInterpolator(coords, image,
                                                        method="linear",
                                                        bounds_error=False,
                                                        fill_value=bg_val,
                                                )

            # Get random elastic deformations
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                                mode="constant", cval=0.) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                                mode="constant", cval=0.) * alpha
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                                mode="constant", cval=0.) * alpha

            # Define sample points
            x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
            indices = np.reshape(x + dx, (-1, 1)), \
                     np.reshape(y + dy, (-1, 1)), \
                     np.reshape(z + dz, (-1, 1))

            # Interpolate 3D image image
            #image = np.empty(shape=image.shape, dtype=dtype)
            image = im_intrps(indices).reshape(shape)

            # Interpolate labels

            lab_intrp = RegularGridInterpolator(coords, labels,
                                               method="nearest",
                                               bounds_error=False,
                                               fill_value=0,
                                                )

            labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)
            #print('变换了')
            return image, labels
        return image, labels

    def __call__(self, img, msk):
        img = img.numpy()
        msk = msk.numpy()
        img = img.squeeze(0)
        msk = msk.squeeze(0)
        prob = (random.uniform(0, 1))
        img, msk = self._elastic_transform(img, msk, prob=prob)
        img = torch.FloatTensor(img).unsqueeze(0)  # 在第一维增加一个维度
        msk = msk.astype(float)
        msk = torch.FloatTensor(msk).unsqueeze(0)
        #print(img.shape, msk.shape)
        return img, msk



def random_noise(img_numpy, mean=0, std=0.001):
    noise = np.random.normal(mean, std, img_numpy.shape)

    return img_numpy + noise


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.001):
        self.mean = mean
        self.std = std

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """

        return random_noise(img_numpy, self.mean, self.std), label

# 三个轴随机翻转，没有概率
class RandomFlip:
    def _random_flip(self, img, label):
        axes = [0, 1, 2]
        rand = np.random.randint(0, 3)
        img = self._flip_axis(img, axes[rand])
        img = np.squeeze(img)
        label = self._flip_axis(label, axes[rand])
        label = np.squeeze(label)
        return img, label

    def _flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def __call__(self, img, msk):
        img = img.numpy()
        msk = msk.numpy()
        img = img.squeeze(0)
        msk = msk.squeeze(0)
        img, msk =  self._random_flip(img, msk)
        img = torch.FloatTensor(img).unsqueeze(0)  # 在第一维增加一个维度
        msk = msk.astype(float)
        msk = torch.FloatTensor(msk).unsqueeze(0)
        # print(img.shape, msk.shape)
        return img, msk
# -----------------------------------------------------------------------


class Resize:
    def __init__(self, scale):
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        self.scale = scale

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=(1,self.scale,self.scale),mode='trilinear', align_corners=False, recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)
        return img[0], mask[0]

class RandomResize:
    def __init__(self,s_rank, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0],self.w_rank[1])
        random_h = random.randint(self.h_rank[0],self.h_rank[1])
        random_s = random.randint(self.s_rank[0],self.s_rank[1])
        self.shape = [random_s,random_h,random_w]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape,mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].long()

class RandomCrop:
    def __init__(self, slices):
        self.slices =  slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):

        ss, es = self._get_range(mask.size(1), self.slices)
        
        # print(self.shape, img.shape, mask.shape)
        tmp_img = torch.zeros((img.size(0), self.slices, img.size(2), img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), self.slices, mask.size(2), mask.size(3)))
        tmp_img[:,:es-ss] = img[:,ss:es]
        tmp_mask[:,:es-ss] = mask[:,ss:es]
        return tmp_img, tmp_mask

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        print(self._flip(img, prob).shape, self._flip(mask, prob).shape)
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)


class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  # base默认取16，因为4次下采样后为1
        self.max_size = max_size 
        if self.max_size%self.base:
            self.max_size = self.max_size - self.max_size%self.base # max_size为限制最大采样slices数，防止显存溢出，同时也应为16的倍数
    def __call__(self, img , label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1)//2 - slice_num//2
        right =  img.size(1)//2 + slice_num//2

        crop_img = img[:,left:right]
        crop_label = label[:,left:right]
        return crop_img, crop_label

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask