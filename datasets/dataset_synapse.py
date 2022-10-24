import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import json
import torchvision.transforms as T
from .aug import RandomAffine, GaussianBlur, To_PIL_Image,JointCompose, JointTo_Tensor

def normalize(img, mean, std):
        mean = torch.as_tensor(mean,dtype=img.dtype,device=img.device)
        std = torch.as_tensor(std,dtype=img.dtype,device=img.device)
        return (img-mean)/std

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample

class resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image = normalize(image, mean=[52.95], std=[52.15])
        # print(image.shape)
        # quit()
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class AcdcDataset(Dataset):
    target_augment: object

    def __init__(self, base_dir, list_dir = None, split='train', transform = None):
        self.transform = transform
        self.split = split

        with open(os.path.join(base_dir, self.split+'.json'), 'r') as f:
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        img = h5py.File(self.data_infos[index], 'r')['image']
        gt = h5py.File(self.data_infos[index], 'r')['label']
        # print(np.unique(gt))
        # img = np.array(img)[:, :, None].astype(np.float32)
        # gt = np.array(gt)[:, :, None].astype(np.float32)
        #
        img = np.array(img)[:, :]
        gt = np.array(gt)[:, :]
        # print(np.unique(gt))
        img_id = self.data_infos[index].split("_set/P_")[1].split(".hdf5")[0]
        # print(img_id)
        # exit()

        sample = {'image': img, 'label': gt, 'case_name': img_id}
        if self.transform:
            sample = self.transform(sample)
        return sample
