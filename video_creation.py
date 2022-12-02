import cv2
import os
results_folder = 'results_11_30_2022_22_36_33'
results_path = os.path.abspath(os.path.join(results_folder))
video_path = os.path.join(results_path, 'video')

images = [img for img in os.listdir(video_path) if img.endswith(".png")]
frame = cv2.imread(os.path.join(video_path, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(os.path.join(video_path, 'performance.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

for image in images:
    I = cv2.imread(os.path.join(video_path, image))
    video.write(I)

cv2.destroyAllWindows()
video.release()