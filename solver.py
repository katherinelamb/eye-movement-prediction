import custom_modules as cm
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

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.

data_train = None # = TensorDataset('(*tensors)')
loader_train = None #DataLoader(data_train, batch_size=64, 
                     #     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

data_val = None #TensorDataset('(*tensors)')
loader_val = None # DataLoader(data_val, batch_size=64,                        # amount remaining
                   #     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 1200)))

data_test = None #TensorDataset('(*tensors)')
loader_test = None #DataLoader(data_test, batch_size=64)


USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train/validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



def pretrain(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    NUM_TRAIN = 49000

    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by subtracting the mean RGB value and dividing by the
    # standard deviation of each RGB value; we've hardcoded the mean and std.
    transform = T.Compose([
                    T.Resize((64,64)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    # We set up a Dataset object for each split (train / val / test); Datasets load
    # training examples one at a time, so we wrap each Dataset in a DataLoader which
    # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
    # training set into train and val sets by passing a Sampler object to the
    # DataLoader telling how it should sample from the underlying Dataset.
    cifar10_train = dset.CIFAR10('./pretrain_datasets/CIFAR', train=True, download=True,
                                transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./pretrain_datasets/CIFAR', train=True, download=True,
                            transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./pretrain_datasets/CIFAR', train=False, download=True, 
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
#                 check_accuracy_part34(loader_train, model)
                check_accuracy(loader_val, model)
                print()