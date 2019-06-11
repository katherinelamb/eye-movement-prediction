import custom_modules as cm
import gaze_dataset as gdata
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


def check_gaze_accuracy(loader, name_of_set, model, mini):
    if loader.dataset.train:
        print('Checking accuracy on', name_of_set, 'set')
    else:
        print('Checking accuracy on test set')   
    total_percentage_points = 0
    total_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    W = 4 if mini else 64
    with torch.no_grad():
        for sample_batched in loader:
            x = sample_batched['image']
            batch_size = x.shape[0]
            y = torch.zeros((batch_size, W, W))
            coords = sample_batched['coords'].squeeze()
            # print ('coords', coords.shape, coords)
            idx = torch.arange(0, batch_size, out=torch.LongTensor())
            y[idx, coords[:,0], coords[:,1]] += 1
            y = y.unsqueeze(1)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            # scores = model(x)
            # sums = torch.sum(scores, dim=(2,3), keepdim=True)
            # percentages = scores / sums
            percentages = model(x)
            # maxes = torch.max(percentages, 0)
            # guesses = torch.where(percentages == maxes, 1, 0)
           # guesses = torch.argmax(percentages, dim=0) wrong
            reshaped = percentages.view(batch_size, -1).argmax(1).view(-1, 1)
            guesses = torch.cat((reshaped // W, reshaped % W), dim=1)
            guesses.unsqueeze(1)
            print ('guesses', guesses.shape)
            print ('y', y.shape)
            # print ('percentages', percentages)
            percentages_of_correct_pixels = percentages * y
            # print ('percentages', percentages_of_correct_pixels)
            # total_correct += torch.sum(guesses * y)
            # print ('guesses[:,0]', guesses[:,0])
            # print ('guesses[:,1]', guesses[:,1])
            # print ('indexed into y', y[idx, 0, guesses[:,0], guesses[:,1]])
            total_correct += torch.sum(y[idx, 0, guesses[:,0], guesses[:,1]])
            # print ('correct so far', total_correct)
            # print ('out of:', torch.sum(y))
            total_percentage_points += torch.sum(percentages_of_correct_pixels)
            num_samples += batch_size
        perc_acc = float(total_percentage_points) / num_samples
        guess_acc = float(total_correct) / num_samples
        print('percentage points correct: (%.2f)' % (100 * perc_acc))
        print('percent guesses correct: (%.2f)' % (100 * guess_acc))


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

def train(model, optimizer, mini=True, epochs=1):
    '''
    Train full Seccade model on our gathered data
    mini: flag for guessing 4x4 sections of image rather than 16x16
    '''
    NUM_TRAIN = 8000
    NUM_DEV = 9000
    DATA_TOTAL = 10000
    BATCH_SIZE = 64
    data_path = './'
    labels_path = './labels.csv'
    printed_y_once = False
    # hopefully this CIFAR norm and std generalizes to our data, if not, may switch to imagenet
    transform = T.Compose([
                    gdata.ToTensor(),
                    gdata.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), mini),
                ])
    dataset = gdata.GazeDataset(labels_path, data_path, train=True, transform=transform, mini=mini)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))#, num_workers=2)
    dev_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_DEV)))#, num_workers=2)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_DEV, DATA_TOTAL)))
    # for batch_idx, (data, target) in enumerate(load_dataset(data_path, transform)):
    print ('train')
    W = 64
    if mini:
        W = 4
    for e in range(epochs):
        if e > 0:
            printed_y_once = True
        for t, sample_batched in enumerate(train_loader):
            print ('iter', t)
            x = sample_batched['image']
            batch_size = x.shape[0]
            y = torch.zeros((batch_size, W, W))
            coords = sample_batched['coords'].squeeze()
            # print ('coords', coords.shape, coords)
            idx = torch.arange(0, batch_size, out=torch.LongTensor())
            y[idx, coords[:,0], coords[:,1]] += 1
            if not printed_y_once:
                print ('sum of y', torch.sum(y, dim=0))
            y = y.unsqueeze(1)
            # model.train()  # put model to training mode ###can't do with pretrained cuz puts all parts back
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)#dtype=torch.long)
            percentages = model(x)
            
            # sums = torch.sum(scores, dim=(2,3), keepdim=True)
            # percentages = scores / sums
            # percentages = softmax(scores)
            
            # print ('scores', scores.shape)
            # print ('y', y.shape)
            # print ('scores', scores)
            # print ('percentages', percentages)
            loss = F.binary_cross_entropy(percentages, y)

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
                # check_gaze_accuracy(dev_loader, 'val', model, mini)
                # exit()
    check_gaze_accuracy(dev_loader, 'val', model, mini)
    compute_train = input('compute train accuracy? (y/n): ')
    if compute_train.lower()[0] == 'y':
        check_gaze_accuracy(train_loader, 'train,', model, mini)
    
    
    
    
    # uncomment to test result of loading data
    # for batch_idx, sample_batched in enumerate(train_loader):
    #     print (batch_idx)
    #     print ('data.size', sample_batched['image'].size())
    #     print ('coords size', sample_batched['coords'].size())
    # print ('dev')
    # for batch_idx, sample_batched in enumerate(dev_loader):
    #     print (batch_idx)
    #     print ('data.size', sample_batched['image'].size())
    #     print ('coords size', sample_batched['coords'].size())
    #     print ('data.shape', sample_batched['image'].shape)
    #     print ('data', sample_batched['image'])
    # print ('done')



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
                check_accuracy(loader_val, 'val', model)
                # check_accuracy_part34(loader_train, 'train', model)
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