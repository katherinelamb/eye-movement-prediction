import custom_modules as cm
import pandas
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
SAVE_PATH = './saved_models/'


def check_accuracy(loader, name_of_set, model):
    if loader.dataset.train:
        print('Checking accuracy on', name_of_set, 'set')
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


def load_gaze_dataset(data_path, transform):
    
    print (data_path)
    train_dataset = dset.ImageFolder(
        root=data_path,
        transform=transform
    )
    return train_dataset

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     num_workers=0#,
    #     # shuffle=True
    # )
    # return train_loader

def train(model, optimizer, epochs=1):
    '''
    Train full Seccade model on our gathered data
    '''
    data_path = './gaze_train/training_data_singles/overfit_test/'
    labels_path = './gaze_train/labels.csv'
    labels = pandas.read_csv(labels_path)
    # hopefully this CIFAR stuff generalizes to our data, if not, may fix later
    transform = T.Compose([
                    T.Resize((64,64)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    # for batch_idx, (data, target) in enumerate(load_dataset(data_path, transform)):
    for batch_idx, data, target in enumerate(load_gaze_dataset(data_path, transform)):
        pass
    print('TODO: Train here')



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
#                 check_accuracy_part34(loader_train, 'train', model)
                check_accuracy(loader_val, 'val', model)
                print()


def save_model(model, model_name, save_as_single_model=False):
    if not os.path.exists(os.path.dirname(SAVE_PATH+model_name)):
        try:
            os.makedirs(os.path.dirname(SAVE_PATH+model_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if save_as_single_model:
        torch.save(model.state_dict(), SAVE_PATH+model_name)
    else:
        for idx, submodel in enumerate(model.children()):
            print (idx, submodel)
            torch.save(submodel.state_dict(), SAVE_PATH+model_name+str(idx))

def check_pretrain_model_acc_from_file(model_name, check_train, check_val):
    model = cm.PretrainModel()
    for idx, submodel in enumerate(model.children()):
        dirname = SAVE_PATH+model_name+str(idx)
        print (idx, submodel)
        if os.path.exists(os.path.dirname(dirname)):
            try:
                submodel.load_state_dict(torch.load(dirname))
            except OSError as exc:
                print ('skipping idx:', idx)
                continue
    
    NUM_TRAIN = 49000

    transform = T.Compose([
                    T.Resize((64,64)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    
    if check_train:
        cifar10_train = dset.CIFAR10('./pretrain_datasets/CIFAR', train=True, download=True,
                                    transform=transform)
        loader_train = DataLoader(cifar10_train, batch_size=64, 
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
        check_accuracy(loader_train, 'train', model)
    if check_val:
        cifar10_val = dset.CIFAR10('./pretrain_datasets/CIFAR', train=True, download=True,
                                transform=transform)
        loader_val = DataLoader(cifar10_val, batch_size=64, 
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
        check_accuracy(loader_val, 'val', model)