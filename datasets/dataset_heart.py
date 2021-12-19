import os
import random
import h5py
import numpy as np
import torch
from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import cv2
from torch.utils.data import Dataset
import os


def random_rot_flip(image, label):
#     k = np.random.randint(0, 4)
    k = np.random.choice([0, 2])
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, angle):
    angle = np.random.randint(-angle, angle)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, Hflip=None, Vflip=None, rot90=None, angle=None, train=True):
        self.output_size = output_size
        self.Hflip = Hflip
        self.Vflip = Vflip
        self.rot90 = rot90
        self.angle = angle
        self.train = train
    
    def __str__(self):
        
        if self.train:
            return f'Hflip, Vflip, rot90_02, rotate{self.angle}, zoom'
        else:
            return f'Hflip_{self.Hflip}, Vflip_{self.Vflip}, angle_{self.angle}, zoom'
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if self.train:
            if random.random() > 0.5:
                image, mask = random_rot_flip(image, mask)
            elif random.random() > 0.5:
                image, mask = random_rotate(image, mask, self.angle)
        else:  # for tta
            if self.Hflip is not None:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if self.Vflip is not None:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if self.rot90 is not None:
                image = np.rot90(image, self.rot90)
                mask = np.rot90(mask, self.rot90)
            if self.angle is not None:
                image = ndimage.rotate(image, self.angle, order=0, reshape=False)
                mask = ndimage.rotate(mask, self.angle, order=0, reshape=False)
        
        x, y, _ = image.shape # h,w
        if x != self.output_size[0] or y != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_CUBIC) # cv -> (w,h)
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'image': image, 'mask': mask.long()}
        
        return sample


class Heart_dataset(Dataset):
    def __init__(self, base_dir, transform=None, infer=False):
        self.data_dir = Path(base_dir)
        self.transform = transform  # using transform in torch!
        self.sample_list = sorted([splitext(file)[0] for file in listdir(self.data_dir) if file.endswith('.png')])
        self.infer = infer

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        data_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, data_name)
        image = Image.open(data_path + '.png').convert('RGB')
        image = np.asarray(image)
        if not self.infer:
            label = np.load(data_path + '.npy')
        else:
            label = np.zeros_like(image[:,:,0]) # dummy
        # h crop
#         print(image.shape, data_path, np.where(image == 0))
#         if np.min(image) == 1:
#             image = image - 1 # (1,255) to (0,254)
#         start = np.where(image == 0)[0][0]
#         image = image[start:]
#         label = label[start:]
        
        sample = {'image': image / 255, 'mask': label}
        
        if self.transform:
            sample = self.transform(sample)
            
            sample["image"] = sample["image"].float()
            sample["mask"] = sample["mask"].long()
        
        sample['file_name'] = data_name

        return sample
