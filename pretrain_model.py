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
    SAVE_PATH = './saved_models/'
    load_model = True
    model_name = argv[1]
    learning_rate = 3e-3
    ################################################################################
    # Instantiate model and a corresponding optimizer #
    ################################################################################
    model = cm.PretrainModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    solver.pretrain(model, optimizer)

    if not os.path.exists(os.path.dirname(SAVE_PATH+model_name)):
        try:
            os.makedirs(os.path.dirname(SAVE_PATH+model_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    torch.save(model.state_dict(), SAVE_PATH+model_name)



if __name__ == "__main__":
    try:
        assert (len(sys.argv) == 2)
    except:
        print('arguments:', sys.argv[1:])
        print ('must include name of model as argume t')
        exit()

    main(sys.argv)
