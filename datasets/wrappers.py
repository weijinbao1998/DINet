import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples
from utils import make_coord


def to_tensor(data):
    return torch.from_numpy(data)



def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))



def to_rgb_pixel_samples(img, flatten=True):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:], flatten=flatten)
    if flatten:
        rgb = img.view(3, -1).permute(1, 0)
    else:
        rgb = img
    return coord, rgb





@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr, img_ref = self.dataset[idx]
        # print(img_lr.shape,img_hr.shape,img_ref.shape)
        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale

        h_lr, w_lr = img_lr.shape[-2:]
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        img_ref = img_ref[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr, crop_ref = img_lr, img_hr, img_ref
        crop_ref_lr = resize_fn(img_ref, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            ###### Tar #######
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            ###### Ref #######
            crop_ref_lr = augment(crop_ref_lr)
            crop_ref = augment(crop_ref)

        ###### Tar #######
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        ###### Ref #######
        _, ref_rgb = to_pixel_samples(crop_ref.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            ###### Tar #######
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            ###### Ref #######
            ref_rgb = ref_rgb[sample_lst]

        ref_w = int(np.sqrt(ref_rgb.shape[0]))
        ref_c = ref_rgb.shape[1]
        # prit(ref_w,ref_c)
        ref_hr = ref_rgb.contiguous().view(ref_c, ref_w, ref_w)
        ###### Tar #######
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        return {
            'inp': crop_lr,
            'inp_hr_coord': hr_coord,
            'inp_cell': cell,
            'ref': crop_ref_lr,
            'ref_hr': ref_hr,
            'gt': hr_rgb
        }


def add_gaussian_noise(img_lr, sigma=15):
    """
    给图像添加高斯噪声，保证噪声对原图影响较小。

    参数：
    - img_lr (Tensor): 输入图像，形状为 (C, H, W)，数据类型为 torch.float32，像素值范围 [0, 1]。
    - sigma (float): 高斯噪声的标准差，默认为 0.05。较小的噪声标准差避免影响图像内容。

    返回：
    - Tensor: 加噪后的图像。
    """
    # 生成与图像相同大小的高斯噪声
    noise = (torch.randn_like(img_lr) * sigma)/255.0
    print(noise.min(), noise.max())
    # 加上噪声
    img_noisy = img_lr + noise

    # 确保噪声图像的像素值在合理范围 [0, 1]
    img_noisy = torch.clamp(img_noisy, 0.0, 1.0)

    return img_noisy

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.inp_size is None:
            return self._process_as_paired(idx)

        else:
            return self._process_as_downsampled(idx)

    def _process_as_paired(self, idx):
        img_hr, img_ref = self.dataset[idx]
        s = self.scale_max

        if s%2 != 0:
            img_lr = resize_fn(img_hr, round(img_hr.shape[-2] // s))
            h_lr, w_lr = img_lr.shape[-2:]
            # img_hr = img_hr[:, :h_hr, :w_hr]
            # img_ref = img_ref[:, :h_hr, :w_hr]
        else:
            img_lr = resize_fn(img_hr, img_hr.shape[-2] // s)
            h_lr, w_lr = img_lr.shape[-2:]
            # img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            # img_ref = img_ref[:, :h_lr * s, :w_lr * s]

        add_noise = False
        if add_noise:
            sigma = 10
            img_lr = add_gaussian_noise(img_lr,sigma)


        # img_lr = resize_fn(img_hr, round(img_hr.shape[-2] // s))
        # h_lr, w_lr = img_lr.shape[-2:]
        # img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        # img_ref = img_ref[:, :h_lr * s, :w_lr * s]

        crop_lr, crop_hr, crop_ref = img_lr, img_hr, img_ref
        crop_ref_lr = resize_fn(img_ref, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_ref_lr = augment(crop_ref_lr)
            crop_ref = augment(crop_ref)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        _, ref_rgb = to_pixel_samples(crop_ref.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            ref_rgb = ref_rgb[sample_lst]

        ref_w = int(np.sqrt(ref_rgb.shape[0]))
        ref_c = ref_rgb.shape[1]
        ref_hr = ref_rgb.contiguous().view(ref_c, ref_w, ref_w)

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        # print(crop_lr.shape,ref_hr.shape,crop_ref_lr.shape)
        return {
            'inp': crop_lr,
            'inp_hr_coord': hr_coord,
            'inp_cell': cell,
            'ref': crop_ref_lr,
            'ref_hr': ref_hr,
            'gt': hr_rgb
        }

    def _process_as_downsampled(self, idx):
        T2_img, T1_img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, T2_img.shape[-2] - w_hr)
        y0 = random.randint(0, T2_img.shape[-1] - w_hr)

        T2_crop_hr = T2_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T2_crop_lr = resize_fn(T2_crop_hr, w_lr)

        T1_crop_hr = T1_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T1_crop_lr = resize_fn(T1_crop_hr, w_lr)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            T2_crop_lr = augment(T2_crop_lr)
            T2_crop_hr = augment(T2_crop_hr)
            T1_crop_lr = augment(T1_crop_lr)
            T1_crop_hr = augment(T1_crop_hr)



        T2_hr_coord, T2_hr_rgb = to_pixel_samples(T2_crop_hr.contiguous())
        _, T1_hr_rgb = to_pixel_samples(T1_crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(T2_hr_coord), self.sample_q, replace=False)
            T2_hr_coord = T2_hr_coord[sample_lst]
            T2_hr_rgb = T2_hr_rgb[sample_lst]
            T1_hr_rgb = T1_hr_rgb[sample_lst]

        ref_w = int(np.sqrt(T1_hr_rgb.shape[0]))
        ref_c = T1_hr_rgb.shape[1]
        T1_ref_hr = T1_hr_rgb.view(ref_c, ref_w, ref_w)
        T2_cell = torch.ones_like(T2_hr_coord)
        T2_cell[:, 0] *= 2 / T2_crop_hr.shape[-2]
        T2_cell[:, 1] *= 2 / T2_crop_hr.shape[-1]
        s_formatted = "{:.2f}".format(s)
        s_tensor = torch.tensor(float(s_formatted))
        return {
            'inp': T2_crop_lr,
            'inp_hr_coord': T2_hr_coord,
            'inp_cell': T2_cell,
            'ref': T1_crop_lr,
            'ref_hr': T1_ref_hr,
            'gt': T2_hr_rgb,
            'scale': s_tensor,
        }

'''@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        T2_img, T1_img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        # if self.inp_size != None:
        #     s = random.uniform(self.scale_min, self.scale_max)
        # else:
        #     s = self.scale_max
        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, T2_img.shape[-2] - w_hr)
        y0 = random.randint(0, T2_img.shape[-1] - w_hr)

        ####### prepare inp #########
        T2_crop_hr = T2_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T2_crop_lr = resize_fn(T2_crop_hr, w_lr)
        ####### prepare ref #########
        T1_crop_hr = T1_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T1_crop_lr = resize_fn(T1_crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            ####### prepare inp #########
            T2_crop_lr = augment(T2_crop_lr)
            T2_crop_hr = augment(T2_crop_hr)
            ####### prepare ref #########
            T1_crop_lr = augment(T1_crop_lr)
            T1_crop_hr = augment(T1_crop_hr)

        # print(T2_crop_hr.shape,T2_crop_lr.shape)
        ####### prepare inp #########
        T2_hr_coord, T2_hr_rgb = to_pixel_samples(T2_crop_hr.contiguous())

        ####### prepare ref #########
        _, T1_hr_rgb = to_pixel_samples(T1_crop_hr.contiguous())
        # print(T2_hr_coord.shape,T2_hr_rgb.shape)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(T2_hr_coord), self.sample_q, replace=False)
            ####### prepare inp #########
            T2_hr_coord = T2_hr_coord[sample_lst]
            T2_hr_rgb = T2_hr_rgb[sample_lst]
            ####### prepare ref #########
            T1_hr_rgb = T1_hr_rgb[sample_lst]

        ref_w = int(np.sqrt(T1_hr_rgb.shape[0]))
        ref_c = T1_hr_rgb.shape[1]
        T1_ref_hr = T1_hr_rgb.view(ref_c, ref_w, ref_w)
        ####### prepare inp #########
        T2_cell = torch.ones_like(T2_hr_coord)
        T2_cell[:, 0] *= 2 / T2_crop_hr.shape[-2]
        T2_cell[:, 1] *= 2 / T2_crop_hr.shape[-1]
        return {
            'inp': T2_crop_lr,
            'inp_hr_coord': T2_hr_coord,
            'inp_cell': T2_cell,
            'ref': T1_crop_lr,
            'ref_hr': T1_ref_hr,
            'gt': T2_hr_rgb,
        }'''