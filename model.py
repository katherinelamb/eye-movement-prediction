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
import solver
import sys
import os
import numpy as np

def main(argv):
    '''
    Pass in a model name, either to load or to save to.
    All models will exist in the saved_models folder
    pass in flag '-t' if you want to train a new model
    pass in flag '-tp' if you want to train a model, but load
    SingleCropEncoder weights which have a head start from pretraining
    '''
    SAVE_PATH = './saved_models/'
    load_model = True
    partial_load = False
    model_name = argv[1]
    if len(argv) > 2:
        if argv[len(argv)-1] == '-t':
            load_model = False
        if argv[len(argv)-1] == '-tp':
            print ('attempting load')
            partial_load = True
    # learning_rate = 3e-3
    learning_rate = 3e3

    model = None
    optimizer = None
    ################################################################################
    # Instantiate model and a corresponding optimizer #
    ################################################################################
    model = cm.SeccadeModel()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if not load_model:
        solver.train(model, optimizer)
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
        solver.save_model(model, model_name)

    if load_model:
        print ('loading')
        single_crop_encoder_idx = '0'
        for idx, submodel in enumerate(model.children()):
            dirname = None
            if idx < 3 and partial_load:
                dirname = SAVE_PATH+model_name+single_crop_encoder_idx
            else:
                dirname = SAVE_PATH+model_name+str(idx)
            print (idx, submodel)
            print (dirname)
            if os.path.exists(os.path.dirname(dirname)):
                try:
                    submodel.load_state_dict(torch.load(dirname))
                    submodel.eval()
                except OSError as exc:
                    print ('skipping idx:', idx)
                    continue
        solver.train(model, optimizer)
        # N,C,H,W = 4, 3, 64, 64
        # in1 = torch.rand((N,C,H,W))
        # in2 = torch.rand((N,C,H,W))
        # in3 = torch.rand((N,C,H,W))
        # output = model(in1, in2, in3)
        # print('output', output.shape)




if __name__ == "__main__":
    try:
        assert (len(sys.argv) >= 2)
    except:
        print('arguments:', sys.argv[1:])
        print ('must include name of model as first argument')
        exit()

    main(sys.argv)
