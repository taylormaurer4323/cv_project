# -*- coding: utf-8 -*-
'''
GPU Training of Mobile Stereo Net to perform high-performing stereo disparity estimation. Note:
    you will need this repo:
        https://github.com/taylormaurer4323/mobilestereonet
'''
#General imports
import os
import sys
import gc
import json
import time
import matplotlib.pyplot as plt
import numpy as np

#Deep Learning stuff
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn


#################################################################################
## Datasetup #####################################################################
#################################################################################
if __name__ == '__main__':
    #Some parallel friendly imports
    fpath = os.path.abspath(os.path.join(os.getcwd()))
    fpath_mobilenet = os.path.join(fpath, 'mobilestereonet')
    sys.path.append(fpath)
    sys.path.append(fpath_mobilenet)
    # Argoverse
    from argoverse_light.camera_stats import RECTIFIED_STEREO_CAMERA_LIST
    from argoverse_light.stereo_dataloader import ArgoverseStereoDataLoader
    from argoverse_light.eval_utils import compute_disparity_error_image

    # Mobilenet
    # Add mobilenet path

    from mobilestereonet.models import __models__, model_loss
    from mobilestereonet.utils import *
    from mobilestereonet.utils.KittiColormap import *
    from argoverse_dataset import ArgoverseDataset
    import mobilenet_trainer as mnt


    if torch.cuda.is_available():
        print('We have GPU!')
    else:
        print('We do NOT have GPU :(')

    #Data directory locations:
    data_dir = os.path.join('..','..', 'shared', 'data')
    stereo_img_dir = os.path.join(data_dir, 'rectified_stereo_images_v1.1')
    disparity_img_dir = os.path.join(data_dir, 'disparity_maps_v1.1')

    #summary logger:
    logdir = '.'
    logger = SummaryWriter(logdir)

    max_disp = 192



    cudnn.benchmark = True
    ran_seed = 1
    torch.manual_seed(ran_seed)
    torch.cuda.manual_seed(ran_seed)

    split_name = "train"
    log_ids = ['08a8b7f0-c317-3bdb-b3dc-b7c9b6d033e2', '0ef28d5c-ae34-370b-99e7-6709e1c4b929', '10b3a1d8-e56c-38be-aaf7-ef2f862a5c4e',
    '3138907e-1f8a-362f-8f3d-773f795a0d01',  'c6911883-1843-3727-8eaa-41dc8cda8993',
    '3d20ae25-5b29-320d-8bae-f03e9dc177b9',  'cd38ac0b-c5a6-3743-a148-f4f7b804ed17',
    '4137e94a-c5da-38bd-ad06-6d57b24bccd0',  'd4d9e91f-0f8e-334d-bd0e-0d062467308a',
    '45753856-4575-4575-4575-345754906624',  'dcdcd8b3-0ba1-3218-b2ea-7bb965aad3f0',
    '52af191b-ba56-326c-b569-e37790db40f3',  'de777454-df62-3d5a-a1ce-2edb5e5d4922',
    '53037376-5303-5303-5303-553038557184l',  'e9bb51af-1112-34c2-be3e-7ebe826649b4',
    '53213cf0-540b-3b5a-9900-d24d1d41bda0',  'ebe7a98b-d383-343b-96d6-9e681e2c6a36',
    '577ea60d-7cc0-34a4-a8ff-0401e5ab9c62',  'f0826a9f-f46e-3c27-97af-87a77f7899cd',
    '5c251c22-11b2-3278-835c-0cf3cdee3f44',  'f3fb839e-0aa2-342b-81c3-312b80be44f9',
    '649750f3-0163-34eb-a102-7aaf5384eaec',  'fa0b626f-03df-35a0-8447-021088814b8b',
    '64c12551-adb9-36e3-a0c1-e43a0e9f3845']
    #dataset used by tensor flow
    argoverse_ds = ArgoverseDataset(data_dir, True, split_name, log_ids=log_ids)
    TrainImgLoader = DataLoader(argoverse_ds, 1, shuffle=False, num_workers = 4)


    split_name = "val"
    val_log_ids = \
    ["00c561b9-2057-358d-82c6-5b06d76cebcf",
     "033669d3-3d6b-3d3d-bd93-7985d86653ea",
     "1d676737-4110-3f7e-bec0-0c90f74c248f",
     "2d12da1d-5238-3870-bfbc-b281d5e8c1a1",
     "33737504-3373-3373-3373-633738571776",
     "39556000-3955-3955-3955-039557148672"]

    ''' "5ab2697b-6e3e-3454-a36a-aba2c6f27818",
    "64724064-6472-6472-6472-764725145600",
    "6db21fda-80cd-3f85-b4a7-0aadeb14724d",
    "70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c",
    "85bc130b-97ae-37fb-a129-4fc07c80cca7",
    "cb0cba51-dfaf-34e9-a0c2-d931404c3dd8",
    "cb762bb1-7ce1-3ba5-b53d-13c159b532c8",
    "da734d26-8229-383f-b685-8086e58d1e05",
    "e9a96218-365b-3ecd-a800-ed2c4c306c78",
    "f1008c18-e76e-3c24-adcc-da9858fac145",
    "f9fa3960-537f-3151-a1a3-37a9c0d6d7f7"]'''
    argoverse_ds_test = ArgoverseDataset(data_dir, False, split_name, log_ids=val_log_ids)
    print(len(argoverse_ds_test), 'total data logs within validation set')
    TestImgLoader = DataLoader(argoverse_ds_test, 1, shuffle=False, num_workers = 4)

    #################################################################################
    ## TRAINING #####################################################################
    #################################################################################

    #get the model:
    model = __models__['MSNet2D'](max_disp)
    model = nn.DataParallel(model)
    model.cuda()
    prev_checkpoint = 'archive_checkpoints/checkpoint_000097_downsampled_dilated.ckpt'
    state_dict = torch.load(prev_checkpoint)
    model.load_state_dict(state_dict['model'])


    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (.9, 0.999))

    start_epoch = 0
    training_epochs = 300
    lrepochs = "4:10"
    summary_freq = 10
    save_freq = 1

    mnt.train(model, optimizer, TrainImgLoader, logger, logdir, start_epoch = start_epoch,
              epochs = training_epochs, lrepochs= lrepochs, learning_rate = learning_rate, summary_freq = summary_freq,
              max_disp = 192, save_freq = save_freq, TestImgLoader=TestImgLoader)

# def train(model, optimizer, TrainImgLoader, logger, logdir, start_epoch = 0, epochs = 1, lrepochs= "200:10", summary_freq = 10, 
#     train_limit = 100000, max_disp = 192, save_freq = 1, TestImgLoader=0):