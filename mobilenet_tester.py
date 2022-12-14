# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:43:38 2022

@author: taylo
"""
from mobilestereonet.models import __models__, model_loss
from mobilestereonet.utils import *
import time
import os
import torch
import gc
from utils.KittiColormap import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import platform
import torchvision
import torchvision.transforms as transforms
from mobilestereonet.utils import *

def test(model, img_loader, test_iteration_limit = 100000):
    time_start = time.time()
    print("Generating the disparity maps...")
    timestamp = time.strftime("%m_%d_%Y_%H_%M_%S")
    folder_name = "results_"+timestamp
    results_path = os.path.join("./"+folder_name)
    pred_path = os.path.join(results_path, "predictions")
    truth_path = os.path.join(results_path, "ground_truth", "stereo_front_left_rect_disparity")
    truth_obj_path = os.path.join(results_path, "ground_truth", "stereo_front_left_rect_objects_disparity")
    ref_img_path = os.path.join(results_path, "reference_images")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(truth_path, exist_ok=True)
    os.makedirs(truth_obj_path, exist_ok=True)
    os.makedirs(ref_img_path, exist_ok=True)
    agg_time = 0
    for batch_idx, sample in enumerate(img_loader):

        print('Batch ', batch_idx, 'out of ', len(img_loader))
        
        disp_est_tn, inf_time = test_sample(model, sample)
        agg_time = agg_time + inf_time
        print('Inference time: ', inf_time, 'running average: ', agg_time/(batch_idx+1), 'total run-time: ', time.time() - time_start)
        disp_est_np = tensor2numpy(disp_est_tn)
        #These should be zero...
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        disp_truth_np = tensor2numpy(sample["disparity"])
        left_truth_np = tensor2numpy(sample["left_truth_img"])
        disp_truth_obj_np = tensor2numpy(sample["disparity_obj"])

        left_filenames = sample["left_filename"]

        
        for disp_est, disp_truth, disp_truth_obj, top_pad, right_pad, fn, left_img in zip(disp_est_np, disp_truth_np, disp_truth_obj_np, top_pad_np, right_pad_np, left_filenames, left_truth_np):
            #print('[result] disp_est shape, ', disp_est.shape, 'disp_truth_size', disp_truth.shape, 'left_img shape', left_img.shape)

            assert len(disp_est.shape) == 2                        
            disp_est[disp_est < 0] = 0
            #continue to convert??
            disp_est = np.uint16(disp_est*256)#the otehr version has a *256
            disp_truth = np.uint16(disp_truth*256)
            disp_truth_obj = np.uint16(disp_truth*256)

            truth_img = np.uint8(left_img.transpose((1,2,0))*255)

            if platform == "linux" or platform == "linux2":
                name = fn.split('/')
            elif platform == "win32" or platform == "win64":
                name = fn.split('\\')

            name = name[-1].split('.')
            pred_name = 'pred'+name[0]+'.png'
            truth_name = 'truth'+name[0]+'.png'
            truth_obj_name = 'truth_obj'+name[0]+'.png'
            img_name = 'img'+name[0]+'.png'

            pred_fn = os.path.join(pred_path, pred_name)
            truth_fn = os.path.join(truth_path, truth_name)
            truth_obj_fn = os.path.join(truth_obj_path, truth_obj_name)
            img_fn = os.path.join(ref_img_path, img_name)
            
            #disp_est = kitti_colormap(disp_est)
            cv2.imwrite(pred_fn, disp_est)
            cv2.imwrite(truth_fn, disp_truth)
            cv2.imwrite(truth_obj_fn, disp_truth_obj)
            cv2.imwrite(img_fn, truth_img)
            


        if batch_idx >= test_iteration_limit:
            break
    
    print('Total time: ', time.time() - time_start )
    return results_path, pred_path, truth_path, truth_obj_path, ref_img_path

@make_nograd_func
def test_sample(model, sample):

    model.eval()
    if len(sample['left'].shape) == 3:
        #reshape:
        c, h, w = sample['left'].shape
        if torch.cuda.is_available():
            left_img = torch.reshape(sample['left'], (1, c,h,w))
            left_img.cuda()
            right_img = torch.reshape(sample['right'], (1, c,h,w))
            right_img.cuda()
        else:
            left_img = torch.reshape(sample['left'], (1, c, h, w))
            right_img = torch.reshape(sample['right'], (1, c, h, w))
    else:
        if torch.cuda.is_available():
            left_img = sample['left'].cuda()
            right_img = sample['right'].cuda()
        else:
            left_img = sample['left']
            right_img = sample['right']
    istart = time.time()
    disp_ests = model(left_img, right_img)
    inf_time = time.time() - istart
    return disp_ests[-1], inf_time

@make_nograd_func
def test_sample_mb(model, sample):

    model.eval()
    h,w =sample[0].size
    c=3

    if torch.cuda.is_available():
        left_img = sample[0].resize((1456, 960))

        left_img = torchvision.transforms.functional.pil_to_tensor(left_img)
        left_img = torch.reshape(left_img, (1, c, 960, 1456))
        left_img = left_img.float()
        left_img.cuda()
        right_img = sample[0].resize((1456, 960))
        right_img = torchvision.transforms.functional.pil_to_tensor(right_img)
        right_img = right_img.float()
        right_img = torch.reshape(right_img, (1, c, 960, 1456))
        right_img.cuda()
    else:
        left_img = torchvision.transforms.functional.pil_to_tensor(sample[0])
        left_img = torch.reshape(left_img, (1, c, h, w))
        right_img = torchvision.transforms.functional.pil_to_tensor(sample[1])
        right_img = torch.reshape(right_img, (1, c, h, w))

    istart = time.time()
    disp_ests = model(left_img, right_img)
    inf_time = time.time() - istart
    return disp_ests[-1], inf_time


@make_nograd_func
def test_single_instance(model, index, dataset, default_size=(512, 960), bFlip_to_depth=False, focal_length=0, baseline=.2986, bMiddlebury=False):
    # Grab thing:
    sample = dataset.__getitem__(index)
    if bMiddlebury:
        disp_est, inference_time = test_sample_mb(model, sample)
    else:
        disp_est, inference_time = test_sample(model, sample)
    print(sample['left_filename'])
    disp_est_np = np.reshape(tensor2numpy(disp_est), default_size)
    disp_est_np = np.float32(disp_est_np)
    disp_est_np[disp_est_np < 0] = 0

    if bFlip_to_depth:
        valid_pixels = disp_est_np > 0
        disp_est_np = np.float32((focal_length * baseline) / (disp_est_np + (1.0 - valid_pixels)))

    # Dilate map:
    #map_dil = cv2.dilate(disp_est_np, kernel=np.ones((2, 2), np.uint8), iterations=7)
    li = sample["left_truth_img"].permute(1, 2, 0).numpy()
    ri = sample["right_truth_img"].permute(1, 2, 0).numpy()
    map_true = sample["disparity"]
    #map_dil_true = cv2.dilate(map_true, kernel=np.ones((2, 2), np.uint8), iterations=7)

    plt.figure(figsize=(24, 18), dpi=150)
    plt.subplot(2, 2, 1)
    plt.imshow(li)
    plt.subplot(2, 2, 2)
    plt.imshow(ri)
    plt.subplot(2,2,3)
    plt.imshow(disp_est_np)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2, 2, 4)
    plt.imshow(map_true)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    return disp_est_np, map_true
@make_nograd_func
def test_single_instance_mb(model, index, dataset, default_size=(1456, 960), bFlip_to_depth=False, focal_length=0, baseline=.2986):
    # Grab thing:
    sample = dataset.__getitem__(index)

    disp_est, inference_time = test_sample_mb(model, sample)

    disp_est_np = np.reshape(tensor2numpy(disp_est), (960, 1456))
    disp_est_np = np.float32(disp_est_np)
    disp_est_np[disp_est_np < 0] = 0

    if bFlip_to_depth:
        valid_pixels = disp_est_np > 0
        disp_est_np = np.float32((focal_length * baseline) / (disp_est_np + (1.0 - valid_pixels)))

    # Dilate map:
    #map_dil = cv2.dilate(disp_est_np, kernel=np.ones((2, 2), np.uint8), iterations=7)
    li = sample[0]
    ri = sample[1]
    map_true = sample[3]
    map_true = cv2.resize(map_true.astype(float), (1456, 960))
    #map_dil_true = cv2.dilate(map_true, kernel=np.ones((2, 2), np.uint8), iterations=7)

    plt.figure(figsize=(24, 18), dpi=150)
    plt.subplot(2, 2, 1)
    plt.imshow(li)
    plt.subplot(2, 2, 2)
    plt.imshow(ri)
    plt.subplot(2,2,3)
    plt.imshow(disp_est_np)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2, 2, 4)
    plt.imshow(map_true)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

    return disp_est_np, map_true
if __name__ == '__main__':
    test()