import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import sys
import os
import numpy as np


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class SingleCropEncoder(nn.Module):
    '''
    input should be 
    '''
    def __init__(self, model_load_path=None, in_channel=3, channel_1=16, channel_2=16, channel_3=16):
        super().__init__()
        ########################################################################
        # Set up the layers needed for a encoding each view                    #
        ########################################################################

        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1, bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if model_load_path is None:
            # initialize for fresh train
            nn.init.kaiming_normal_(self.conv1.weight)
            nn.init.constant_(self.conv1.bias, 0)
            nn.init.kaiming_normal_(self.conv2.weight)
            nn.init.constant_(self.conv2.bias, 0)
            nn.init.kaiming_normal_(self.conv3.weight)
            nn.init.constant_(self.conv3.bias, 0)
        else:
            self.conv1.load_state_dict(torch.load(model_load_path))
            self.conv2.load_state_dict(torch.load(model_load_path))
            self.conv3.load_state_dict(torch.load(model_load_path))

    def forward(self, x):
        # X starts (N,C,64,64)      (C = 3 ->RGB)
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)        # 32x32
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)        # 16x16
        x = F.relu(self.conv3(x))
        encoding = self.max_pool(x) # 8x8
        return encoding

class ThreeLayerConvTransposeNet(nn.Module):
    '''
    input should be 
    '''
    def __init__(self, in_channel=48, channel_1=16, channel_2=1):
        super().__init__()
        ########################################################################
        # Set up the layers needed for a encoding each view                    #
        ########################################################################

        self.convT1 = nn.ConvTranspose2d(in_channel, channel_1, kernel_size=4, stride=2, padding=1, bias=True)
        nn.init.kaiming_normal_(self.convT1.weight)
        nn.init.constant_(self.convT1.bias, 0)
        self.convT2 = nn.ConvTranspose2d(channel_1, channel_2, kernel_size=6, stride=4, padding=1, bias=True)
        nn.init.kaiming_normal_(self.convT2.weight)
        nn.init.constant_(self.convT2.bias, 0)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # X starts (N,C,8,8)      (C = 48 = 3*16)
        x = F.relu(self.convT1(x))   # (N,16,16,16)
        x = self.convT2(x)
        decoding = self.softmax(x)   # (N,1,64,64)
        return decoding

class TwoLayerConvNet(nn.Module):
    '''
    take three crop encodings and shrink to single 4x4 area to guess eye location
    '''
    def __init__(self, in_channel=48, channel_1=8, channel_2=1):
        super().__init__()
        ########################################################################
        # Set up the layers needed for a encoding each view                    #
        ########################################################################

        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=2)
        # initialize
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        # X starts (N,C,8,8)      (C = 48 = 3*16)
        N = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)        # 4x4
        x = self.conv2(x)
        print ('decoding pre-softmax', x)
        # decoding = self.softmax(x) # 4x4 percentages
        decoding = self.softmax(x.view(*x.size()[:2],-1)).view_as(x)
        assert (decoding.shape == (N, 1, 4,4))
        print ('decoding', decoding)
        # print ('sum', torch.sum(decoding, dim=(2,3)))
        print ('sums of pixels over all examples\n', torch.sum(decoding, dim=(0,1)))
        # exit()
        # exit()
        return decoding

class PostVggDecoder(nn.Module):
    '''
    take three crop encodings and shrink to single 4x4 area to guess eye location
    '''
    def __init__(self, in_channel=512*3, channel_1=512, channel_2=128, channel_3=32, channel_4=1):
        super().__init__()
        ########################################################################
        # Set up the layers needed for a encoding each view                    #
        ########################################################################

        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=3, padding=1, bias=True)
        self.conv1_1 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1, bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(channel_3, channel_4, kernel_size=3, padding=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        # initialize
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        nn.init.constant_(self.conv1_1.bias, 0)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.constant_(self.conv1_2.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x128, x256, x512):
        # each x starts (N,C,8,8)      (C = 512)
        encodings = torch.cat((x128, x256, x512), 1) #concatenate along channel axis
        N = encodings.shape[0]
        x = F.relu(self.conv1(encodings))
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.max_pool(x)        # 4x4
        x = self.conv2(x)
        print ('decoding pre-softmax', x)
        # decoding = self.softmax(x) # 4x4 percentages
        decoding = self.softmax(x.view(*x.size()[:2],-1)).view_as(x)
        assert (decoding.shape == (N, 1, 4,4))
        print ('decoding', decoding)
        # print ('sum', torch.sum(decoding, dim=(2,3)))
        print ('sums of pixels over all examples\n', torch.sum(decoding, dim=(0,1)))
        # exit()
        return decoding

class SeccadeModel(nn.Module):
    def __init__(self, name=None, version=2):
        super().__init__()
        # all SingleCropEncoders take 64x64 images
        # number after sc means original size of image before resizing
        self.sc128 = SingleCropEncoder(name)
        self.sc256 = SingleCropEncoder(name)
        self.sc512 =  SingleCropEncoder(name)
        self.decoder = None
        if version == 1:
            convT_net = ThreeLayerConvTransposeNet()
            self.decoder = convT_net # 64x64 outpuy
        else:
            conv_net = TwoLayerConvNet()
            self.decoder = conv_net # 4x4 output
        
        
    def forward(self, x): # x is shape (batch, channel, H, W)
        # forward encodes images as (N,16,8,8) tensors
        #split crops
        W = 64
        x128 = x[:,:,:, :W]
        x256 = x[:,:,:, W : 2*W]
        x512 = x[:,:,:, 2*W : 3*W]
        x128 = self.sc128.forward(x128)
        x256 = self.sc128.forward(x256)
        x512 = self.sc128.forward(x512)

        # get (N,48,8,8) encoding of all three views
        encodings = torch.cat((x128, x256, x512), 1) #concatenate along channel axis
        
        # decode and get distribution over pixels of guess of seccade destination
        decoded = self.decoder.forward(encodings)
        return decoded

class PretrainModel(nn.Module):
    '''pretrains on CIFAR10'''
    def __init__(self):
        super().__init__()
        # all SingleCropEncoders take 64x64 images
        # number after sc means original size of image before resizing
        self.encoder = SingleCropEncoder()
        self.fc = nn.Linear(16*8*8, 10, bias=True)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        # encode images as (N,16,8,8) tensors
        encoding = self.encoder(x) # equivalent of three-layer conv
        encodings= flatten(encoding)
        # decode and get distribution over pixels of guess of seccade destination
        score = self.fc(encodings)
        return score
