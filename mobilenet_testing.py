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
import torchvision
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
    from argoverse_light.eval import StereoEvaluator


    # Mobilenet
    # Add mobilenet path

    from mobilestereonet.models import __models__, model_loss
    from mobilestereonet.utils import *
    from mobilestereonet.utils.KittiColormap import *
    from argoverse_dataset import ArgoverseDataset
    import mobilenet_tester as mnt


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
    log_ids = ['08a8b7f0-c317-3bdb-b3dc-b7c9b6d033e2']
    # dataset used by tensor flow
    argoverse_ds_train = ArgoverseDataset(data_dir, True, split_name, log_ids=log_ids)
    TrainImgLoader = DataLoader(argoverse_ds_train, 1, shuffle=False, num_workers=4)


    split_name = "val"
    val_log_ids = \
    ["5ab2697b-6e3e-3454-a36a-aba2c6f27818",
    "64724064-6472-6472-6472-764725145600",
    "6db21fda-80cd-3f85-b4a7-0aadeb14724d",
    "70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c",
    "85bc130b-97ae-37fb-a129-4fc07c80cca7",
    "cb0cba51-dfaf-34e9-a0c2-d931404c3dd8",
    "cb762bb1-7ce1-3ba5-b53d-13c159b532c8",
    "da734d26-8229-383f-b685-8086e58d1e05",
    "e9a96218-365b-3ecd-a800-ed2c4c306c78",
    "f1008c18-e76e-3c24-adcc-da9858fac145",
    "f9fa3960-537f-3151-a1a3-37a9c0d6d7f7"]
    val_log_ids = ["00c561b9-2057-358d-82c6-5b06d76cebcf"]
    argoverse_ds_test = ArgoverseDataset(data_dir, False, split_name, log_ids=val_log_ids)
    print(len(argoverse_ds_test), 'total data logs within validation set')
    TestImgLoader = DataLoader(argoverse_ds_test, 1, shuffle=False, num_workers = 4)


    middlebury_data = torchvision.datasets.Middlebury2014Stereo(data_dir)
    mb_data_loader = DataLoader(middlebury_data,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=4)

    #################################################################################
    ## TESTING #####################################################################
    #################################################################################

    #get the model:
    model = __models__['MSNet2D'](max_disp)
    model = nn.DataParallel(model)
    model.cuda()
    from pathlib import Path

    #load the best checkpoint!MSNet2D_DS_KITTI2015
    state_dict = torch.load('checkpoint_000097.ckpt')
    model.load_state_dict(state_dict['model'])
    bSingleTest= False
    #315969643874312720
    if bSingleTest:
        #pred, truth = mnt.test_single_instance_mb(model, 0, mb_data_loader.dataset, default_size=(1456, 960))
        pred, truth = mnt.test_single_instance(model, 72, argoverse_ds_test)
    else:
        results_path, pred_path, truth_path, truth_obj_path, ref_img_path = mnt.test(model, TestImgLoader)
        # Create save dir for disparity error images:
        disp_err_path = os.path.join(results_path, 'disparity_err_img')
        os.makedirs(disp_err_path, exist_ok=True)
        # Convert to aboslute paths:
        pred_path = Path(os.path.abspath(pred_path))
        outer_truth_path = Path(os.path.abspath(os.path.join(truth_path, '..')))
        disp_err_path = Path(os.path.abspath(disp_err_path))
        ref_img_path = Path(os.path.abspath(ref_img_path))
        results_path = Path(os.path.abspath(results_path))

        # Create stereo evaluator:
        evaluator = StereoEvaluator(
            pred_path,
            outer_truth_path,
            disp_err_path,
            save_disparity_error_image=True,
            num_procs=1
        )
        # Evaluate:
        metrics = evaluator.evaluate()

        print(metrics[0])
        ground_truth_path = os.path.join(outer_truth_path, 'stereo_front_left_rect_disparity')
        disp_err_path = os.path.join(disp_err_path, 'ground_truth')
        #import cv2
        #import os
        image_folder = 'images'
        video_name = 'video.avi'
        pred_image_names = [img for img in os.listdir(pred_path)]
        disp_err_names = [img for img in os.listdir(disp_err_path)]
        ground_truth_names = [img for img in os.listdir(ground_truth_path)]
        ref_img_names = [img for img in os.listdir(ref_img_path)]
        video_path = os.path.join(results_path, 'video')
        os.makedirs(video_path)
        for i, img_name in enumerate(pred_image_names):
            print('index: ', i)
            pred_img = cv2.imread(os.path.join(pred_path, pred_image_names[i]))
            disp_err_img = cv2.imread(os.path.join(disp_err_path, disp_err_names[i]))
            outer_truth_img= cv2.imread(os.path.join(ground_truth_path, ground_truth_names[i]))
            ref_img = cv2.imread(os.path.join(ref_img_path, ref_img_names[i]))
            plt.subplot(2,2,1)
            plt.imshow(pred_img)
            plt.title('Predicted')
            plt.subplot(2,2,2)
            plt.imshow(outer_truth_img)
            plt.title('Ground Truth')
            plt.subplot(2,2,3)
            plt.imshow(ref_img)
            plt.title('Reference Image')
            plt.subplot(2,2,4)
            plt.imshow(disp_err_img)
            plt.title('Disparity Error')
            plt.savefig(os.path.join(video_path,"file%02d.png" % i))



        images = [img for img in os.listdir(video_path) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(video_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join(video_path, 'performance.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

        for image in images:
            I = cv2.imread(os.path.join(video_path, image))
            video.write(I)

        cv2.destroyAllWindows()
        video.release()
