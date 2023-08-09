import os
from typing import Tuple, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tempo3.data.pdfs import p_uni

import h5py

class VideoDatasetH5(Dataset):
    """
    Dataset used for Tempo ss-training.
    """

    def __init__(
            self, 
            path, 
            transform=None, 
            proximity:int=3, 
            pdf:Callable[..., int]=p_uni
        ) -> None:
        """
        Args:
            path (str): Path to video frames,
            transform: Transformations applied to frames,
            proximity: Proximity parameter dependent on pdf (tau for uniform, sigma for normal),
            pdf: Probability density function used for sampling f_j 
        """
        
        self.p = proximity
        self.transform = transform
        self.pdf = pdf

        self.file = h5py.File(path, "r")

    def __len__(self) -> int:
        return len(self.file["frames"])

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns samples of form (f_i, f_j, i, j) where f_i is the ith frame of the video.
        """

        # 1. get one element x
        # image = Image.open(self.image_paths[index])
        image = torch.as_tensor(self.file["frames"][index])

        # 2. sample element x' in the neighbourhood of x
        pb = self.pdf(index, self.p, self.__len__())
        index_d = np.random.choice(np.arange(len(pb)), p=pb)
        # image_d = Image.open(self.image_paths[index_d])
        image_d = torch.as_tensor(self.file["frames"][index_d])

        if self.transform:
            image = self.transform(image)
            image_d = self.transform(image_d)

        # 3. return (x, x')
        return (image, image_d, torch.tensor(0), torch.tensor(0))

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    train_dataset = VideoDatasetH5('./datasets/hand_2', transform=transform, proximity=30, train=True)
    test_dataset = VideoDatasetH5('./datasets/hand_2', transform=transform, proximity=30, train=False)

    a=0