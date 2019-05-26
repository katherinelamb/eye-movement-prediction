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
import solver
import os
import numpy as np

NUM_TRAIN = 1000 #??????

def main(argv):
    '''
    Pass in a model name, either to load or to save to.
    All models will exist in the saved_models folder
    pass in flag '-t' if you want to train a new model
    '''
    model_name = argv[1]
    learning_rate = 3e-3
    # learning_rate = 4e-4
    ################################################################################
    # Instantiate model and a corresponding optimizer #
    ################################################################################
    model = cm.PretrainModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
    solver.pretrain(model, optimizer, epochs=3)
    solver.save_model(model, model_name)



if __name__ == "__main__":
    try:
        assert (len(sys.argv) == 2)
    except:
        print('arguments:', sys.argv[1:])
        print ('must include name of model as argument')
        exit()

    main(sys.argv)
