import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from simple_detector import *
from advanced_detector import CameraPipeline, undistort_img

from imageio import imwrite


import logging
logging.basicConfig()
format_ = '%(asctime) - %(message)s'
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)

# calibration n test cases
calibration_folder = 'camera_cal'
test_cases = 'test_images'

# test set
from advanced_detector import hls_select
test_set = glob.glob(test_cases + '/*.jpg')


cal_images = glob.glob(calibration_folder + '/*.jpg')

process_cam = CameraPipeline(9,6)
process_cam.calibrate_cam(cal_images)

test_length = len(test_set)
for i in range(0, test_length):
    #logger.info(print(test_pack[i]))
    img_name = test_set[i].split('/')[1]
    
    img, size = read_video(test_set[i])
    result = undistort_img(img, process_cam.objpoints, process_cam.imgpoints)    
    
    # name
    output_name = 'undist_' + img_name
    savepath = os.path.join('output_images', output_name)
    
    # getting seg fault
    imwrite(savepath, result.astype(np.uint8))
