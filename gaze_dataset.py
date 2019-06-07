from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

### basic structure of this class based on Pytorch Custom Dataset Tutorial
class GazeDataset(Dataset):
    """Gaze coords dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coords_frame = pd.read_csv(csv_file)
        print (self.coords_frame)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coords_frame)

    def __getitem__(self, idx):
        name = 'entry' + str(idx) + '.jpg'
        img_name = os.path.join(self.root_dir, name)
                                # self.coords_frame.iloc[idx, 0])
        image = io.imread(img_name)
        coords = self.coords_frame.iloc[idx, 1:][1:-1].as_matrix()
        coords = coords.astype('float').reshape(-1, 2)
        sample = {'image': image, 'coords': coords}

        if self.transform:
            sample = self.transform(sample)

        return sample


def show_coords(image, coords):
    """Show image with coords"""
    plt.imshow(image)
    plt.scatter(coords[:, 0], coords[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


# Let’s instantiate this class and iterate through the data samples. 
# We will print the sizes of first 4 samples and show their coords.

gaze_dataset = GazeDataset(csv_file='./gaze_train/labels.csv',
                                    root_dir='./gaze_train/training_data_singles/overfit_test/')

fig = plt.figure()

for i in range(len(gaze_dataset)):
    sample = gaze_dataset[i]

    print(i, sample['image'].shape, sample['coords'].shape)
    print(sample['coords'])

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_coords(**sample)

    if i == 3:
        plt.show()
        input()
        break