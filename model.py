import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim
from torch.utils.data import ConcatDataSet
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataSet
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

NUM_TRAIN = 1000 #??????

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
data_train = TensorDataSet('(*tensors)')
loader_train = DataLoader(data_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

data_val = TensorDataSet('(*tensors)')
loader_val = DataLoader(data_val, batch_size=64,                        # amount remaining
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 1200)))

data_test = TensorDataSet('(*tensors)')
loader_test = DataLoader(data_test, batch_size=64)


USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)


class SingleCropEncoder(nn.Module):
    '''
    input should be 
    '''
    def __init__(self, in_channel=3, channel_1=16, channel_2=16, channel_3=16):
        super().__init__()
        ########################################################################
        # Set up the layers needed for a encoding each view                    #
        ########################################################################

        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        self.conv3 = nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

    def forward(self, x):
        # X starts 64x64xTxC      (C = 3 ->RGB)
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
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        self.convT2 = nn.ConvTranspose2d(channel_1, channel_2, kernel_size=6, stride=4, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        

    def forward(self, x):
        # X starts Nx8x8xC      (C = 48 = 3*16)
        x = F.relu(self.convT1(x))   #Nx16x16x16
        decoding = F.sigmoid(self.convT2(x))   # Nx64x64x1
        return decoding


class SeccadeModel(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        # all SingleCropEncoders take 64x64 images
        # number after sc means original size of image before resizing
        self.sc128 = SingleCropEncoder()
        self.sc256 = SingleCropEncoder()
        self.sc512 =  SingleCropEncoder()
        self.convT_net = ThreeLayerConvTransposeNet()
        
    def forward(self, x128, x256, x512):
        # encode images as (N,8,8,16) tensors
        x128 = self.sc128.forward(x128)
        x256 = self.sc128.forward(x256)
        x512 = self.sc128.forward(x512)

        # get (N,8x8x48) encoding of all three views
        encodings = torch.cat([x128, x256, x512], axis=3) #concatenate along channel axis
        
        # decode and get distribution over pixels of guess of seccade destination
        decoded_score = self.convT_net.forward(encodings)
        return decoded_score



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



def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
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

##### TRAINING #######

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None
################################################################################
# Instantiate model and a corresponding optimizer #
################################################################################
in_channel = 3
num_classes = 10
model = SeccadeModel(in_channel, channel_1, channel_2, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train(model, optimizer)