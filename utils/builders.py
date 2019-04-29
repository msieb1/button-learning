from __future__ import print_function, division

import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import randn
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# plt.ion()   # interactive mode
from sklearn.preprocessing import OneHotEncoder

def load_encodings(datapath):
    with np.load(datapath) as data:
        colors = data['colors']
        shapes = data['shapes']
    colors_ohe = OneHotEncoder().fit_transform(colors.reshape(-1, 1)).toarray()
    shapes_ohe = OneHotEncoder().fit_transform(shapes.reshape(-1, 1)).toarray()
    ohe = np.hstack([colors_ohe, shapes_ohe])
    return ohe    

def load_encoding_and_button(datapath):
    with np.load(datapath) as data:
        colors = data['colors']
        shapes = data['shapes']
        is_buttons = data['is_buttons']
    colors_ohe = OneHotEncoder().fit_transform(colors.reshape(-1, 1)).toarray()
    shapes_ohe = OneHotEncoder().fit_transform(shapes.reshape(-1, 1)).toarray()
    ohe = np.hstack([colors_ohe, shapes_ohe])        
    return is_buttons, ohe  

class ShapeDataset(Dataset):
    """End effector finger tip dataset."""

    def __init__(self, root_dir, transform=None):
        """Initializes dataset
        
        Parameters
        ----------
        root_dir : str
            path to raw data
        transform : list, optional
            list of transforms (check torchsample subfolder for all available transforms, or use official torch.transforms library) (the default is None, which does not load any transforms)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels = load_encodings(join(root_dir, 'data.npz'))
        

    def __len__(self):
        """Computes length of dataset (number of overall samples)
        
        Returns
        -------
        int 
            number of all samples in dataset
        """
        return len([f for f in os.listdir(self.root_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    def __getitem__(self, idx):
        """Loads an item from the dataset
        
        Parameters
        ----------
        idx : int
            index of current sample
        
        Returns
        -------
        dict
            Returns a datasample dict containing the keys 'image' and 'label', where label has the same dimension as 'image', but 2 channels, one for each finger tip
        """

        img_path = join(self.root_dir, '{0:06d}.png'.format(idx))
        image = io.imread(img_path)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        image = np.transpose(image, (2, 0, 1)) # check whether correct transpose (channels first vs channels last)
        label = self.labels[idx]

        # image = skimage.img_as_float32(image)
        sample = {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(label).float()}

        # Transform image if provided
        if self.transform:
            sample = self.transform(sample)
        return sample

class ButtonDataset(Dataset):
    """End effector finger tip dataset."""

    def __init__(self, root_dir, transform=None):
        """Initializes dataset
        
        Parameters
        ----------
        root_dir : str
            path to raw data
        transform : list, optional
            list of transforms (check torchsample subfolder for all available transforms, or use official torch.transforms library) (the default is None, which does not load any transforms)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels, self.ohe = load_encoding_and_button(join(root_dir, 'data.npz'))
        

    def __len__(self):
        """Computes length of dataset (number of overall samples)
        
        Returns
        -------
        int 
            number of all samples in dataset
        """
        return len([f for f in os.listdir(self.root_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    def __getitem__(self, idx):
        """Loads an item from the dataset
        
        Parameters
        ----------
        idx : int
            index of current sample
        
        Returns
        -------
        dict
            Returns a datasample dict containing the keys 'image' and 'label', where label has the same dimension as 'image', but 2 channels, one for each finger tip
        """

        img_path = join(self.root_dir, '{0:06d}.png'.format(idx))
        image = io.imread(img_path)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        image = np.transpose(image, (2, 0, 1)) # check whether correct transpose (channels first vs channels last)
        label = self.labels[idx]
        encoding = self.ohe[idx]

        image = skimage.img_as_float32(image)
        sample = {'image': torch.from_numpy(image).float(), 'label': torch.Tensor([label]).float(), 'encoding': torch.from_numpy(encoding).float()}

        # Transform image if provided
        if self.transform:
            sample = self.transform(sample)
        return sample

class GaussianAdditiveNoiseTransform(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        im = sample['image']
        noise = self.std * randn(im.shape) + self.mean
        noise_im = torch.clamp(im + noise, 0, 1)

        return {
            k : noise_im if k == 'image' else v 
            for k, v in sample.items()
        }