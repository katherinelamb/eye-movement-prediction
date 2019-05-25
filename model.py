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
    if len(argv) > 2:
        if argv[2] == '-t':
            load_model = False

    learning_rate = 3e-3

    model = None
    optimizer = None
    ################################################################################
    # Instantiate model and a corresponding optimizer #
    ################################################################################
    model = cm.SeccadeModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if not load_model:
        # train(model, optimizer)
        N,C,H,W = 4, 3, 64, 64
        in1 = torch.rand((N,C,H,W))
        in2 = torch.rand((N,C,H,W))
        in3 = torch.rand((N,C,H,W))
        fake_truth = torch.ones((N,C,H,W)) * (1/(H*W))
        print ('fake', fake_truth[0,0])
        output = model(in1, in2, in3)
        loss = torch.sum(fake_truth-output)
        print(output.shape)
        print ('output', output)
        loss.backward()
        optimizer.step()
        print ('trained')

        if not os.path.exists(os.path.dirname(SAVE_PATH+model_name)):
            try:
                os.makedirs(os.path.dirname(SAVE_PATH+model_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(model.state_dict(), SAVE_PATH+model_name)

    if load_model:
        model.load_state_dict(torch.load(SAVE_PATH+model_name))
        N,C,H,W = 4, 3, 64, 64
        in1 = torch.rand((N,C,H,W))
        in2 = torch.rand((N,C,H,W))
        in3 = torch.rand((N,C,H,W))
        output = model(in1, in2, in3)
        print('output', output)


if __name__ == "__main__":
    try:
        assert (len(sys.argv) >= 2)
    except:
        print('arguments:', sys.argv[1:])
        print ('must include name of model as first argument')
        exit()

    main(sys.argv)
