# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:43:38 2022

@author: taylo
"""
from mobilestereonet.models import __models__, model_loss
from mobilestereonet.utils import *
import time
import torch
import gc
from utils.KittiColormap import *


#DEFINE TEST SAMPLE
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
        
        if batch_idx % 5 == 0:
            clear_output(wait=True)
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
            name = fn.split('/')
            name = name[-1].split('.')
            pred_name = 'pred'+name[0]+'.png'
            truth_name = 'truth'+name[0]+'.png'
            truth_obj_name = 'truth_obj'+name[0]+'.png'
            img_name = 'img'+name[0]+'.png'

            pred_fn = os.path.join(pred_path, pred_name)
            truth_fn = os.path.join(truth_path, truth_name)
            truth_obj_fn = os.path.join(truth_obj_path, truth_obj_name)
            img_fn = os.path.join(ref_img_path, img_name)
            
            disp_est = kitti_colormap(disp_est)            
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
            right_img = sample['right'.cuda()]
        else:
            left_img = sample['left']
            right_img = sample['right']
    istart = time.time()
    disp_ests = model(left_img, right_img)
    inf_time = time.time() - istart
    return disp_ests[-1], inf_time