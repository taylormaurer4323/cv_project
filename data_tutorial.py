# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:34:32 2022

@author: taylo
"""


'''
Disclaimer all code and all examples shown here are for the most part taken from argoverse-api
here is the link:
    https://github.com/argoai/argoverse-api
    
You'll also want to make sure you have data which you get from their webiste here:
    https://www.argoverse.org/av1.html#stereo-link
Keep scrolling down to the bottom!

Additionally you want to make sure you have all the below imports installed, additionally
you want the following packages installed (you likely have a good amount already 
installed):
    logging, pathlib, typing, imageio, typing_extensions, pandas, multiprocessing, argparse,
    functools, dataclasses, glob, scipy
    '''
    
from pathlib import Path
import json

import os
from argoverse_light.stereo_dataloader import ArgoverseStereoDataLoader
from argoverse_light.camera_stats import RECTIFIED_STEREO_CAMERA_LIST
from argoverse_light.calibration import get_calibration_config
from argoverse_light.eval import StereoEvaluator

#import open3d as o3d <- this ruined everything for me, bad install....
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
import open3d as o3d

STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]
print(STEREO_FRONT_LEFT_RECT)

#NOTE: YOU WANT HTIS TO BE RELATIVE AT LEAST ON WINDOWS
data_dir = os.path.join('..','..', 'shared', 'data')
stereo_img_dir = os.path.join(data_dir, 'rectified_stereo_images_v1.1')
disparity_img_dir = os.path.join(data_dir, 'disparity_maps_v1.1')

print('Data is in: ', data_dir)


split_name = "train"
log_id = "273c1883-673a-36bf-b124-88311b1a80be"
idx = 34
stereo_data_loader = ArgoverseStereoDataLoader(data_dir, split_name)

# Loading the left rectified stereo image paths for the chosen log.
left_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
    log_id=log_id,
    camera_name=STEREO_FRONT_LEFT_RECT,
)

# Loading the right rectified stereo image paths for the chosen log.
right_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
    log_id=log_id,
    camera_name=STEREO_FRONT_RIGHT_RECT,
)

# Loading the disparity map paths for the chosen log.
disparity_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
    log_id=log_id,
    disparity_name="stereo_front_left_rect_disparity",
)

# Loading the disparity map paths for foreground objects for the chosen log.
disparity_obj_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
    log_id=log_id,
    disparity_name="stereo_front_left_rect_objects_disparity",
)

#Now use those functions there and actually load the data
stereo_front_left_rect_image = stereo_data_loader.get_rectified_stereo_image(left_stereo_img_fpaths[idx])
stereo_front_right_rect_image = stereo_data_loader.get_rectified_stereo_image(right_stereo_img_fpaths[idx])

# Loading the ground-truth disparity maps. 
stereo_front_left_rect_disparity = stereo_data_loader.get_disparity_map(disparity_map_fpaths[idx])
# Loading the ground-truth disparity maps for foreground objects only. 
stereo_front_left_rect_objects_disparity = stereo_data_loader.get_disparity_map(disparity_obj_map_fpaths[idx])

#Look at the data:
stereo_front_left_rect_disparity_dil = cv2.dilate(
    stereo_front_left_rect_disparity, 
    kernel=np.ones((2, 2), np.uint8), 
    iterations=7,
)

stereo_front_left_rect_objects_disparity_dil = cv2.dilate(
    stereo_front_left_rect_objects_disparity,
    kernel=np.ones((2, 2), np.uint8),
    iterations=7,
)


plt.figure(figsize=(9, 9))
plt.subplot(2, 2, 1)
plt.title("Rectified left stereo image")
plt.imshow(stereo_front_left_rect_image)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.title("Rectified right stereo image")
plt.imshow(stereo_front_right_rect_image)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.title("Left disparity map")
plt.imshow(
    stereo_front_left_rect_disparity_dil,
    cmap="nipy_spectral",
    vmin=0,
    vmax=192,
    interpolation="none",
)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.title("Left object disparity map")
plt.imshow(
    stereo_front_left_rect_objects_disparity_dil,
    cmap="nipy_spectral",
    vmin=0,
    vmax=192,
    interpolation="none",
)
plt.axis("off")
plt.tight_layout()


#Recovering calibration info:
# First, we need to load the camera calibration. Specifically, we want the camera intrinsic parameters.
calib_data = stereo_data_loader.get_log_calibration_data(log_id=log_id)
camera_config = get_calibration_config(calib_data, camera_name=STEREO_FRONT_LEFT_RECT)

# Getting the focal lenght and baseline. Note that the baseline is constant for the Argoverse stereo rig setup.
focal_lenght = camera_config.intrinsic[0, 0]  # Focal length in pixels.
BASELINE = 0.2986  # Baseline in meters.

# We consider disparities greater than zero to be valid disparities.
# A zero disparity corresponds to an infinite distance.
valid_pixels = stereo_front_left_rect_disparity > 0

# Using the stereo relationship previsouly described, we can recover the depth map by:
stereo_front_left_rect_depth = \
    np.float32((focal_lenght * BASELINE) / (stereo_front_left_rect_disparity + (1.0 - valid_pixels)))

# Recovering the colorized point cloud using Open3D.
left_image_o3d = o3d.geometry.Image(stereo_front_left_rect_image)
depth_o3d = o3d.geometry.Image(stereo_front_left_rect_depth)
rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
    left_image_o3d, 
    depth_o3d, 
    convert_rgb_to_intensity=False, 
    depth_scale=1.0, 
    depth_trunc=200,
)
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
pinhole_camera_intrinsic.intrinsic_matrix = camera_config.intrinsic[:3, :3]
pinhole_camera_intrinsic.height = camera_config.img_height
pinhole_camera_intrinsic.width = camera_config.img_width
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_o3d, pinhole_camera_intrinsic)

# Showing the colorized point cloud using the interactive Plotly.
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1, color=colors),
            )
    ],
  layout=dict(
      scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
          zaxis=dict(visible=False),
            aspectmode="data",
        )
    ),
)
fig.show() #<- not working on my system...


# Defining the SGM parameters (please check the OpenCV documentation for details).
# We found this parameters empirically and based on the Argoverse Stereo data. 
max_disp = 192
win_size = 10
uniqueness_ratio = 15
speckle_window_size = 200
speckle_range = 2
block_size = 11
P1 = 8 * 3 * win_size ** 2
P2 = 32 * 3 * win_size ** 2

# Defining the Weighted Least Squares (WLS) filter parameters.
lmbda = 0.1
sigma = 1.0

# Defining the SGM left matcher.
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=max_disp,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=max_disp,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# Defining the SGM right matcher needed for the left-right consistency check in the WLS filter.
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Defining the WLS filter.
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Computing the disparity maps.
left_disparity = left_matcher.compute(stereo_front_left_rect_image, stereo_front_right_rect_image)
right_disparity = right_matcher.compute(stereo_front_right_rect_image, stereo_front_left_rect_image)

# Applying the WLS filter.
left_disparity_pred = wls_filter.filter(left_disparity, stereo_front_left_rect_image, None, right_disparity)

# Recovering the disparity map.
# OpenCV produces a disparity map as a signed short obtained by multiplying subpixel shifts with 16.
# To recover the true disparity values, we need to divide the output by 16 and convert to float.
left_disparity_pred = np.float32(left_disparity_pred) / 16.0

# OpenCV will also set negative values for invalid disparities where matches could not be found.
# Here we set all invalid disparities to zero.
left_disparity_pred[left_disparity_pred < 0] = 0




plt.figure(figsize=(9, 9))
plt.subplot(2, 2, 1)
plt.title("Rectified left stereo image")
plt.imshow(stereo_front_left_rect_image)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.title("Rectified right stereo image")
plt.imshow(stereo_front_right_rect_image)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.title("Ground-truth left disparity map")
plt.imshow(
    stereo_front_left_rect_disparity_dil,
    cmap="nipy_spectral",
    vmin=0,
    vmax=192,
    interpolation="none",
)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.title("Estimated left disparity map")
plt.imshow(
    left_disparity_pred, 
    cmap="nipy_spectral", 
    vmin=0, 
    vmax=192, 
    interpolation="none"
)
plt.axis("off")
plt.tight_layout()



# We consider disparities greater than zero to be valid disparities.
# A zero disparity corresponds to an infinite distance.
valid_pixels = left_disparity_pred > 0

# Using the stereo relationship previsouly described, we can recover the predicted depth map by:
left_depth_pred = \
    np.float32((focal_lenght * BASELINE) / (left_disparity_pred + (1.0 - valid_pixels)))

# Recovering the colorized point cloud using Open3D.
#left_image_o3d = o3d.geometry.Image(stereo_front_left_rect_image)
#depth_o3d = o3d.geometry.Image(left_depth_pred)
#rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
#    left_image_o3d, 
#    depth_o3d, 
#    convert_rgb_to_intensity=False, 
#    depth_scale=1.0, 
#    depth_trunc=200,
#)
#pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
#pinhole_camera_intrinsic.intrinsic_matrix = camera_config.intrinsic[:3, :3]
#pinhole_camera_intrinsic.height = camera_config.img_height
#pinhole_camera_intrinsic.width = camera_config.img_width
#pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_o3d, pinhole_camera_intrinsic)

# Showing the colorized point cloud using the interactive Plotly.
#points = np.asarray(pcd.points)
# Randomly sampling indices for faster rendering.
#indices = np.random.randint(len(points), size=100000)  
#points = points[indices]
#colors = np.asarray(pcd.colors)[indices]

#fig = go.Figure(
#     data=[
#         go.Scatter3d(
#             x=points[:, 0],
#             y=points[:, 1],
#             z=points[:, 2],
#             mode="markers",
#             marker=dict(size=1, color=colors),
#         )
#     ],
#     layout=dict(
#         scene=dict(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             zaxis=dict(visible=False),
#             aspectmode="data",
#         ),
#     ),
# )
# fig.show()





# Encoding the real disparity values to an uint16 data format to save as an uint16 PNG file.
left_disparity_pred = np.uint16(left_disparity_pred * 256.0)

timestamp = int(Path(disparity_map_fpaths[idx]).stem.split("_")[-1])

# Change the path to the directory you would like to save the result.
# The log id must be consistent with the stereo images' log id.
save_dir_disp = f"."

# The predicted disparity filename must have the format: 'disparity_[TIMESTAMP OF THE LEFT STEREO IMAGE].png' 
filename = f"disparity_{timestamp}.png"

# Writing the PNG file to disk.
cv2.imwrite(filename, left_disparity_pred)




# Path to the predicted disparity maps.
pred_dir = Path(save_dir_disp)

# Path to the ground-truth disparity maps.
gt_dir = Path(f"{data_dir}/disparity_maps_v1.1/{split_name}/{log_id}")
gt_dir = Path(os.path.join(data_dir, 'disparity_maps_v1.1', 'train', log_id))

# Path to save the disparity error image.
save_figures_dir = Path(os.path.join('save_figs'))
save_figures_dir.mkdir(parents=True, exist_ok=True)

print(pred_dir)
print(gt_dir)

# Creating the stereo evaluator.
print('Creating Stereo Evaluator')
evaluator = StereoEvaluator(
    pred_dir,
    gt_dir,
    save_figures_dir,
    save_disparity_error_image=True,
    num_procs=1,
)
print('Running Stereo Evaluator')
# Running the stereo evaluation.
#metrics = evaluator.evaluate() this fails because I only look at a single value, but will come in handy later

# Printing the quantitative results (using json trick for organized printing).
#print(metrics)