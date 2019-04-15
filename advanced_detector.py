import numpy as np
import cv2
from simple_detector import *
from typing import Tuple
import logging
logger = logging.getLogger(__name__)


def genpoints(image_list: list, nx: int = 9, ny: int = 6) -> Tuple[list, list]:
    """
    finds the chessboard corners in order to work out 
    how to undistort image

    """

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    for idx, fname in enumerate(image_list):
        img, shape = read_video(fname)
        gray = grayscale(img)
        
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

# undistort and transform - need mtx and dist from camera calibration
def undistort_img(img: np.ndarray, objpoints: list, imgpoints: list):
    """
    undistorts images based on object points

    """
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist,  rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# HSV color thresholder
def hls_select(img, thresh=(0, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Choose S channel
    s = hls[:,:,2]
    binary_output = np.zeros_like(s) # placeholder line
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output


def adv_pipeline(path: str, objpoints: list, imgpoints: list) -> np.ndarray:
    """
    advanced processing pipeline

    """
    frame, fr_shape = read_video(path)
    undist = undistort_img(frame, objpoints, imgpoints)
    # use S channel
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    #proc_f = grayscale(S)
    proc_f = gaussian_blur(S, 5)
    proc_f = canny(proc_f, 25, 150)
    roi = region_of_interest(proc_f, np.array([[(0.1*fr_shape[1],fr_shape[0]),
                                            (fr_shape[1]*0.95, fr_shape[0]), 
                                            (0.55*fr_shape[1], 0.75*fr_shape[0]), 
                                            (0.45*fr_shape[1],0.75*fr_shape[0])]], dtype=np.int32))
    proc_f_lines = hough_lines(roi, 2, np.pi/180, 16, 5, 50)
    final_lines = find_lines(proc_f, proc_f_lines)
    output_img = draw_lines(proc_f, final_lines)
    output_img = weighted_img(output_img, undist)
    #logger.debug('outputting: {0}'.format(type(frame)))
    
    #yield output_img
    return output_img


class CameraPipeline(object):

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny

    def calibrate_cam(self, cal_images: str):
        self.objpoints, self.imgpoints = genpoints(cal_images, self.nx, self.ny)

    def process(self, img: str):
        output = adv_pipeline(img, self.objpoints, self.imgpoints)
        return output