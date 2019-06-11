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
import vgg_model as vgg

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
            reshaped = x.view(batch_size, -1.argmax(1))
            guesses = torch.cat(((reshaped / W).view(-1,1), (reshaped % W).view(-1,1)), dim=1)
            guesses.unsqueeze(1)
            # print ('percentages', percentages)
            percentages_of_correct_pixels = percentages * y
            # print ('percentages', percentages_of_correct_pixels)
            # total_correct += torch.sum(guesses * y)
            total_correct += torch.sum(y[guesses])
            total_percentage_points += torch.sum(percentages_of_correct_pixels)
            num_samples += batch_size
        perc_acc = float(total_percentage_points) / num_samples
        guess_acc = float(total_correct) / num_samples
        print('percentage points correct: (%.2f)' % (100 * perc_acc))
        print('percent guesses correct: (%.2f)' % (100 * guess_acc))


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
    vgg_model, vgg_mean, vgg_std = vgg.main()
    vgg_model.eval()
    print ('vgg_model loaded')
    exit()
    transform = T.Compose([
                    gdata.vgg_resize(),
                    gdata.ToTensor(),
                    gdata.Normalize(vgg_mean, vgg_std, mini),
                ])
    dataset = gdata.GazeDataset(labels_path, data_path, train=True, transform=transform, mini=mini)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))#, num_workers=2)
    dev_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_DEV)))#, num_workers=2)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_DEV, DATA_TOTAL)))
    # for batch_idx, (data, target) in enumerate(load_dataset(data_path, transform)):
    print ('train')
    W = 128
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
            model.train()  # put model to training mode
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

def main():
    model = None
    learning_rate = 3e-3
    optimizer = None #optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer)

if __name__=="__main__":
    main()