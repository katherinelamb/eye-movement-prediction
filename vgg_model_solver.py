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
import solver
import threading
import logging

USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
# print_every = 100
print_every = 10

print('using device:', device)
SAVE_PATH = './saved_models/'

def check_gaze_accuracy(loader, name_of_set, vgg_model, model, mini):
    if loader.dataset.train:
        print('Checking accuracy on', name_of_set, 'set')
    else:
        print('Checking accuracy on test set')   
    total_percentage_points = 0
    total_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    final_W = 4 if mini else 64
    with torch.no_grad():
        for sample_batched in loader:
            x = sample_batched['image']
            batch_size, _, H, W = x.shape
            y = torch.zeros((batch_size, final_W, final_W))
            coords = sample_batched['coords'].squeeze()
            # print ('coords', coords.shape, coords)
            idx = torch.arange(0, batch_size, out=torch.LongTensor())
            y[idx, coords[:,0], coords[:,1]] += 1
            y = y.unsqueeze(1)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)#dtype=torch.long)
            x128 = x[:,:,:, :W//3]
            x256 = x[:,:,:, W//3 : 2*W//3]
            x512 = x[:,:,:, 2*W//3 : W]
            # encodings
            print('making first encoding')
            e128 = vgg_model(x128)
            print('making second encoding')
            e256 = vgg_model(x256)
            print('making final encoding')
            e512 = vgg_model(x512)
            model.train()  # put model to training mode
            percentages = model(e128, e256, e512)
            # maxes = torch.max(percentages, 0)
            # guesses = torch.where(percentages == maxes, 1, 0)
           # guesses = torch.argmax(percentages, dim=0) wrong
            reshaped = percentages.view(batch_size, -1).argmax(1).view(-1, 1)
            guesses = torch.cat((reshaped // final_W, reshaped % final_W), dim=1)
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

def crop_through_vgg(vgg_model, crop, encodings, crop_num):
    x = vgg_model(crop)
    print('done with crop', crop_num)
    encodings[crop_num] = x


def train(model, optimizer, mini=True, epochs=1):
    '''
    Train full Seccade model on our gathered data
    mini: flag for guessing 4x4 sections of image rather than 16x16
    '''
    NUM_TRAIN = 8000
    NUM_DEV = 9000
    DATA_TOTAL = 10000
    BATCH_SIZE = 8
    data_path = './'
    labels_path = './labels.csv'
    printed_y_once = False
    vgg_model, vgg_mean, vgg_std = vgg.main()
    vgg_model.eval()
    print ('vgg_model loaded')
    # print ('vgg_mean', vgg_mean)
    # print ('vgg_std', vgg_std)
    transform = T.Compose([
                    gdata.VggResize(),
                    gdata.ToTensor(),
                    gdata.Normalize(vgg_mean, vgg_std, mini),
                ])
    dataset = gdata.GazeDataset(labels_path, data_path, train=True, transform=transform, mini=mini)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))#, num_workers=2)
    dev_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_DEV)))#, num_workers=2)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_DEV, DATA_TOTAL)))
    # for batch_idx, (data, target) in enumerate(load_dataset(data_path, transform)):
    print ('train')
    final_W = 4 if mini else 16
    for e in range(epochs):
        if e > 0:
            printed_y_once = True
        for t, sample_batched in enumerate(train_loader):
            print ('iter', t)
            x = sample_batched['image']
            print ('x', x.shape)
            batch_size, _, H, W = x.shape
            y = torch.zeros((batch_size, final_W, final_W))
            coords = sample_batched['coords'].squeeze()
            # print ('coords', coords.shape, coords)
            idx = torch.arange(0, batch_size, out=torch.LongTensor())
            y[idx, coords[:,0], coords[:,1]] += 1
            if not printed_y_once:
                print ('sum of y', torch.sum(y, dim=0))
            y = y.unsqueeze(1)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)#dtype=torch.long)
            x128 = x[:,:,:, :W//3]
            x256 = x[:,:,:, W//3 : 2*W//3]
            x512 = x[:,:,:, 2*W//3 : W]
            crops = (x128, x256, x512)
            encodings = [None, None, None]
            threads = list()
            for index in range(3):
                logging.info("Main    : create and start thread %d.", index)
                x = threading.Thread(target=crop_through_vgg, args=(vgg_model, crops[index], encodings, index,))
                threads.append(x)
                x.start()
                

            for index, thread in enumerate(threads):
                logging.info("Main    : before joining thread %d.", index)
                thread.join()
                logging.info("Main    : thread %d done", index)
            
            # encodings
            # print('making first encoding')
            # e128 = vgg_model(x128)
            # print('making second encoding')
            # e256 = vgg_model(x256)
            # print('making final encoding')
            # e512 = vgg_model(x512)
            e128 = encodings[0]
            e256 = encodings[1]
            e512 = encodings[2]
            model.train()  # put model to training mode
            percentages = model(e128, e256, e512)
            print ('percentages', percentages.shape)
            print ('y', y.shape)
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
                solver.save_model(model, 'post_vgg_decoder', save_as_single_model=True)
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                # check_gaze_accuracy(dev_loader, 'val', vgg_model, model, mini)
                # exit()
    check_gaze_accuracy(dev_loader, 'val', vgg_model, model, mini)
    # compute_train = input('compute train accuracy? (y/n): ')
    # if compute_train.lower()[0] == 'y':
    #     check_gaze_accuracy(train_loader, 'train,', vgg_model, model, mini)

def main():
    model = cm.PostVggDecoder()
    learning_rate = 3e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer)
    solver.save_model(model, 'post_vgg_decoder', save_as_single_model=True)

if __name__=="__main__":
    main()