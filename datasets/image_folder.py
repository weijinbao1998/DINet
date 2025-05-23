import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio

from datasets import register

def to_tensor(data):
    return torch.from_numpy(data)

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file)))


    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        # print(idx,len(self.files))
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = Image.open(x)

            # 检查图像模式，如果是 RGB 则转换为单通道 'L'
            if img.mode in ['RGB', 'RGBA']:
                img = img.convert('L')  # 转换为灰度图像

            return transforms.ToTensor()(img)
            # return transforms.ToTensor()(Image.open(x).convert('L')) # for medical
            # return transforms.ToTensor()(Image.open(x)) # for natural

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x




@register('paired-image-folders')
class PairedImageFolders(Dataset):

    # def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs)
    def __init__(self, root_path_1, root_path_2, **kwargs):

        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        # self.dataset_3 = ImageFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        # return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx]
        return self.dataset_1[idx], self.dataset_2[idx]


@register('mc-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]