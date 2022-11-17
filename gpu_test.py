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
#import matplotlib.pyplot as plt
import numpy as np

#Deep Learning stuff
import torch
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

#Argoverse
from argoverse_light.camera_stats import RECTIFIED_STEREO_CAMERA_LIST
from argoverse_light.stereo_dataloader import ArgoverseStereoDataLoader
from argoverse_light.eval_utils import compute_disparity_error_image

if torch.cuda.is_available():
    print('We have GPU!')
else:
    print('We do NOT have GPU :(')

#Mobilenet
#Add mobilenet path
fpath = os.path.abspath(os.path.join(os.getcwd()))
fpath_mobilenet = os.path.join(fpath, 'mobilestereonet')
sys.path.append(fpath)
sys.path.append(fpath_mobilenet)
from mobilestereonet.models import __models__, model_loss
from mobilestereonet.utils import *
from mobilestereonet.utils.KittiColormap import *
#from argoverse_dataset import ArgoverseDataset
import mobilenet_trainer as mnt

#need
#cv2
#from tensorboardX import SummaryWriter
#import matplotlib.pyplot as plt
