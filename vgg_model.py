from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import gaze_dataset as gdata
import copy
from tqdm import tqdm
import numpy as np
import sys


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def get_truncated_vgg(cnn, device):


    """
    :param cnn:
    :param classifier:
    :param normalization_mean:
    :param normalization_std:
    :param prim_style_img:
    :param sec_style_img:
    :param device:
    :param prim_heatmap: should be a tensor as in the paper
    :param sec_heatmap: should be a tensor as in the paper

    :param content_layers:
    :param style_layers:
    :return:
    """
    cnn = copy.deepcopy(cnn)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():

        if i == 0:
            """
            Only add these for the first layer
            """
            pass
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        print (name)
        model.add_module(name, layer)

    # now we trim off the layers after the last conv
    # for i in range(len(model) - 1, -1, -1):
    for i in range(len(model)):
        print (model[i])
        if isinstance(model[i], nn.Linear):
            break

    model = model[:(i + 1)]

    return model

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    # import the model from pytorch pretrained models
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # vgg networks are trained on images with each channel normalized by mean [0.485, 0.456, 0.406] and
    # standard deviation [0.229, 0.224, 0.225]. Normalize the image using these values before sending it
    # to the network
    cnn_normalization_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(device)
    cnn_normalization_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(device)
    model = get_truncated_vgg(cnn, device)
    return model, cnn_normalization_mean, cnn_normalization_std
    

if __name__=="__main__":
    main()