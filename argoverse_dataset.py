# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:10:15 2022

@author: taylo
"""
from argoverse_light.camera_stats import RECTIFIED_STEREO_CAMERA_LIST
from torch.utils.data import Dataset
from argoverse_light.stereo_dataloader import ArgoverseStereoDataLoader
from math import floor
import torchvision.transforms as transforms
import random

#BUild dataset class for the translation between mobilenet and argoverse data
STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]

class ArgoverseDataset(Dataset):
    def __init__(self, data_dir, training, split_name='train', log_ids=["15c802a9-0f0e-3c87-b516-a3fa02f1ecb0"]):

        self.stereo_data_loader = ArgoverseStereoDataLoader(data_dir, split_name)
        self.training = training
        self.left_stereo_img_fpaths = []
        self.right_stereo_img_fpaths = []
        self.disparity_map_fpaths = []
        self.disparity_obj_map_fpaths = []
        for log_id in log_ids:
            if len(self.left_stereo_img_fpaths) == 0:
                self.left_stereo_img_fpaths = self.stereo_data_loader.get_ordered_log_stereo_image_fpaths(
                    log_id = log_id,
                    camera_name=STEREO_FRONT_LEFT_RECT
                )
                self.right_stereo_img_fpaths = self.stereo_data_loader.get_ordered_log_stereo_image_fpaths(
                    log_id = log_id, 
                    camera_name = STEREO_FRONT_RIGHT_RECT
                )

                self.disparity_map_fpaths = self.stereo_data_loader.get_ordered_log_disparity_map_fpaths(
                    log_id = log_id,
                    disparity_name="stereo_front_left_rect_disparity"
                )

                self.disparity_obj_map_fpaths = self.stereo_data_loader.get_ordered_log_disparity_map_fpaths(
                    log_id = log_id, 
                    disparity_name = "stereo_front_left_rect_objects_disparity"
                )
            else:
                self.left_stereo_img_fpaths = self.left_stereo_img_fpaths + self.stereo_data_loader.get_ordered_log_stereo_image_fpaths(
                        log_id = log_id,
                        camera_name=STEREO_FRONT_LEFT_RECT
                    )
                
                self.right_stereo_img_fpaths = self.right_stereo_img_fpaths + self.stereo_data_loader.get_ordered_log_stereo_image_fpaths(
                    log_id = log_id, 
                    camera_name = STEREO_FRONT_RIGHT_RECT
                )

                self.disparity_map_fpaths = self.disparity_map_fpaths+ self.stereo_data_loader.get_ordered_log_disparity_map_fpaths(
                    log_id = log_id,
                    disparity_name="stereo_front_left_rect_disparity"
                )

                self.disparity_obj_map_fpaths = self.disparity_obj_map_fpaths + self.stereo_data_loader.get_ordered_log_disparity_map_fpaths(
                    log_id = log_id, 
                    disparity_name = "stereo_front_left_rect_objects_disparity"
                )

    def load_path(self):
        print('loading path')
        #I believe this returns the paths for desired data
        #optional defintion
    def get_left_image(self, index, cropped=False):
        #Assumes that the cropping is consistent with test functionality
        left_img =  self.stereo_data_loader.get_rectified_stereo_image(self.left_stereo_img_fpaths[index])
        if cropped:
            crop_w, crop_h = 960, 512
            h, w , c = left_img.shape
            x_start = floor((w - crop_w)/2)
            y_start = floor((h - crop_h)/2)        
            left_img = left_img[y_start:y_start+crop_h, x_start:x_start+crop_w,:]
        
        return left_img

    def get_right_image(self, index, cropped=False):
        #Assumes that the cropping is consistent with test functionality
        right_img = self.stereo_data_loader.get_rectified_stereo_image(self.right_stereo_img_fpaths[index])
        if cropped:
            crop_w, crop_h = 960, 512
            h, w , c = right_img.shape
            x_start = floor((w - crop_w)/2)
            y_start = floor((h - crop_h)/2)        
            right_img = right_img[y_start:y_start+crop_h, x_start:x_start+crop_w,:]

        return right_img

    def get_disparity_image(self, index, cropped=False):
        stereo_front_left_rect_disparity = self.stereo_data_loader.get_disparity_map(self.disparity_map_fpaths[index])
        if cropped:
            crop_w, crop_h = 960, 512
            h, w  = stereo_front_left_rect_disparity.shape
            x_start = floor((w - crop_w)/2)
            y_start = floor((h - crop_h)/2)
            stereo_front_left_rect_disparity = stereo_front_left_rect_disparity[y_start:y_start+crop_h, x_start:x_start+crop_w]
        return stereo_front_left_rect_disparity


    def load_disp(self):
        #optinoal again, looks like it reads a depth map or disparity map
        #from a pfm (floating point image map file format), then returns that
        print('Loading disparity')
        
    def __len__(self):
        #specifies the length of the dataset
        return len(self.left_stereo_img_fpaths)
    def get_transform():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __getitem__(self, index):
        #required function, expected return format of:
        #dictionary, with keys left (leftimg), right (rightimg),
        #   disparity (disparity), top_pad(seems like always 0), 
        #right_pad (seems like always 0), and left_filename (left image filename)
        left_img = self.stereo_data_loader.get_rectified_stereo_image(self.left_stereo_img_fpaths[index])
        right_img = self.stereo_data_loader.get_rectified_stereo_image(self.right_stereo_img_fpaths[index])
        stereo_front_left_rect_disparity = self.stereo_data_loader.get_disparity_map(self.disparity_map_fpaths[index])
        stereo_front_left_rect_objects_disparity = self.stereo_data_loader.get_disparity_map(self.disparity_obj_map_fpaths[index])

        mean = [0.485, 0.456, 0.406]

        std = [0.229, 0.224, 0.225]
        

        if self.training:
            #Need to crop...
            crop_w, crop_h = 512, 256
            h, w , c = left_img.shape
            #some number b/t 0 and all the way to edge (top or right)
            x1 = random.randint(0, w-crop_w)
            y1 = random.randint(0, h - crop_h)

            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            
            left_img_t = T(left_img)
            left_img_t = transforms.functional.crop(left_img_t, y1, x1, crop_h, crop_w)
            right_img_t = T(right_img)
            right_img_t = transforms.functional.crop(right_img_t, y1, x1, crop_h, crop_w)
            #print('y1: ', y1, 'y1+crop_h', y1+crop_h, 'x1: ', x1, 'x1+crop_w', x1+crop_w)
            stereo_front_left_rect_disparity = stereo_front_left_rect_disparity[y1:y1+crop_h, x1:x1+crop_w]
            
            return {"left": left_img_t,
                "right": right_img_t,
                "disparity": stereo_front_left_rect_disparity}
        else:
            #Need to crop...
            crop_w, crop_h = 960, 512
            h, w , c = left_img.shape

            x_start = floor((w - crop_w)/2)
            y_start = floor((h - crop_h)/2)
            
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

            #center crop:
            left_img_t = T(left_img)
            #print('y_start: ', y_start, 'x_start: ', x_start, 'crop_h: ', crop_h, 'crop_w:', crop_w)
            left_img_t = transforms.functional.crop(left_img_t, y_start, x_start, crop_h, crop_w)
            right_img_t = T(right_img)
            right_img_t = transforms.functional.crop(right_img_t, y_start, x_start, crop_h, crop_w)
            stereo_front_left_rect_disparity = stereo_front_left_rect_disparity[y_start:y_start+crop_h, x_start:x_start+crop_w]
            stereo_front_left_rect_objects_disparity = stereo_front_left_rect_objects_disparity[y_start:y_start+crop_h, x_start:x_start+crop_w]
            tmp_T = transforms.ToTensor()
            left_truth_image = tmp_T(self.get_left_image(index, cropped=True))
            right_truth_image = tmp_T(self.get_right_image(index, cropped=True))
            return {"left": left_img_t,
                "right": right_img_t,
                "disparity": stereo_front_left_rect_disparity,
                "disparity_obj": stereo_front_left_rect_objects_disparity,
                "top_pad": 0,
                "right_pad":0,
                "left_filename": self.left_stereo_img_fpaths[index],
                "left_truth_img": left_truth_image,
                "right_truth_img": right_truth_image}