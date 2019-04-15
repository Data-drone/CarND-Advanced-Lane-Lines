#
#
#   Converting the simple detector into an application instead
#
#
#

import cv2
import numpy as np
from typing import Tuple
import logging
logger = logging.getLogger(__name__)


def read_video(vid_path: str) -> Tuple[np.ndarray, np.ndarray] :
    """
    reads video and images and spits out arrays as rgb
    vid_path is the path to a file
    """

    cap = cv2.VideoCapture(vid_path)

    while(True):
        logger.debug('loaded {0}: '.format(vid_path))
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_shape = rgb.shape
        return rgb, frame_shape
    
    return


def grayscale(img: np.ndarray) -> np.ndarray:
    """
    grayscales rgb images
    """

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """
    applies canny edge detector to image
    """

    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    applies a guassian blur
    """

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def find_lines(img: np.ndarray, lines: list) -> list:
    """
    takes the output of hough lines and tries to find the right lines to draw for the lane boundary
    """
    l_x, l_y, r_x, r_y = img.shape[1],0,0,0
    l_x_mid, l_y_mid, r_x_mid, r_y_mid = 0,img.shape[0],img.shape[1],img.shape[0]
    l_grad, r_grad = -0.7,0.7 # default left and right gradients

    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient = np.round(((y2-y1)/(x2-x1)), decimals = 2)
            length = np.round(np.sqrt(np.square(y2-y1) + np.square(x2-x1)), decimals = 2)
            
            # right line
            if np.absolute(gradient) > 0.55 and np.absolute(gradient) < 0.8:
                if gradient > 0: 
                    if x2 > r_x:
                        r_x = x2
                    if y2 > r_y:
                        r_y = y2
                    r_grad = gradient
                    
                    if x1 < r_x_mid:
                        r_x_mid = x1
                    if y1 < r_y_mid:
                        r_y_mid = y1
                        
                # left line
                elif gradient < 0: 
                    # find bottum left corner
                    if x1 < l_x:
                        l_x = x1
                    if y1 > l_y:
                        l_y = y1
                    l_grad = gradient
                    
                    # find top right bit
                    if x2 > l_x_mid:
                        l_x_mid = x2
                    if y2 < l_y_mid:
                        l_y_mid = y2

    if r_y < 0.95*img.shape[0]:
        x_extrap = (img.shape[0]-r_y)/r_grad + r_x
        r_x = int(x_extrap)
        r_y = int(img.shape[0])
    
    # extra fill line
    if l_y < 0.95*img.shape[0]:
        l_x_extrap = (img.shape[0]-l_y)/l_grad + l_x # check
        l_x = int(l_x_extrap)
        l_y = int(img.shape[0])

    # output coordinary and gradient array
    output = [[l_x, l_y, l_x_mid, l_y_mid], [r_x, r_y, r_x_mid, r_y_mid]]

    return output

def hough_lines(img: np.ndarray, rho: int, theta: int, threshold: int, min_line_len: int, max_line_grap: int) -> list:
    """
    run hough lines on image
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_grap)

    return lines

def draw_lines(img: np.ndarray, lines: list, color: list = [255,0,0], thickness: int=2) -> np.ndarray:
    """
    receive lines then draw them
    lines is of format [x1,y1,x2,y2]
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    
    return line_img

def weighted_img(img: np.ndarray, initial_img: np.ndarray, α:float=0.8, β:float=1., γ:float=0.):
    """
    merge lines and original image
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


