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
    from mobilestereonet.utils import *

    for thing in argoverse_ds:
        img_left = thing['left']
        img_right = thing['right']
        img_disp = thing['disparity']
        plt.subplot(1,3,1)
        plt.imshow(img_left.permute(1,2,0))
        plt.subplot(1,3,2)
        plt.imshow(img_right.permute(1,2,0))
        plt.subplot(1,3,3)
        #tmp_map = img_disp.permute(1,2,0).numpy()

        plt.imshow(cv2.dilate(img_disp, kernel=np.ones((2, 2), np.uint8), iterations=7))
        plt.colorbar()
        plt.show()


