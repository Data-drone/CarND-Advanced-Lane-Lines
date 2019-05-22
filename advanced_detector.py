import numpy as np
import cv2
from simple_detector import *
from typing import Tuple
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt


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


def adv_pipeline(path: str, objpoints: list, imgpoints: list, hfd: int = 0.65) -> np.ndarray:
    """
    pipeline for determining polygon dimension

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
                                            (0.55*fr_shape[1], hfd*fr_shape[0]), 
                                            (0.45*fr_shape[1],hfd*fr_shape[0])]], dtype=np.int32))
    proc_f_lines = hough_lines(roi, 4, np.pi/180, 16, 5, 50)
    if proc_f_lines is not None:
        final_lines = find_lines(proc_f, proc_f_lines)
        corner_points = find_points_for_transform(final_lines, undist)

        output_img = draw_lines(proc_f, final_lines)
        output_img = weighted_img(output_img, undist)
    else:
        output_img = frame

    #yield output_img
    return output_img, corner_points

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def search_around_poly(binary_warped: np.ndarray, left_fit: np.ndarray, 
    right_fit: np.ndarray, margin: int=100):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray):
    """

    binary_warped: a perspective shifted image to run the line detector on

    reads in the image and works out the left and right lines

    returns left_fit and right_fit lines and the fitted coordinates
    """
    
    # Find our lane pixels first from the image
    # haven't included code that will research if search fails yet
    if (left_fit is None or right_fit is None):
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty, out_img = search_around_poly(binary_warped, left_fit, right_fit)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def calc_bias(img, left_fit_pts, right_fit_pts):
    middle = img.shape[1]/2

    lane_middle = (right_fit_pts - left_fit_pts)/2 + left_fit_pts
    unscaled_bias = lane_middle - middle
    bias = ((unscaled_bias)*(3.7/700))[0]

    return bias

def plot_points(left_fit_pts, right_fit_pts, ploty):
    """
    orders the points to plotPoly
    """
    left = list(zip(left_fit_pts, ploty))
    right = list(zip(right_fit_pts, ploty))
    left = np.array(left).tolist()
    right = np.array(right).tolist()

    result = []
    result.append(left.pop(0))
    result.extend(right)
    result.extend(left[::-1])
    
    return result


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



def filter_image(frame):
    """

    read an image and apply the filtering needed to create an image to feed into the lane detector

    """

    # create the colour channel image
    R = frame[:,:, 0]

    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    V = hsv[:,:,2]

    thresh_array = np.array([[0, 70, 70], [50,255,255]])
    binary_v = np.zeros_like(R)
    binary_v[(hsv[:,:,0] > thresh_array[0][0]) & (hsv[:,:,0] <= thresh_array[1][0])
            & (hsv[:,:,1] > thresh_array[0][1]) & (hsv[:,:,1] <= thresh_array[1][1])
            & (hsv[:,:,2] > thresh_array[0][2]) & (hsv[:,:,2] <= thresh_array[1][2])] = 1
    
    sobel_grad = dir_threshold(frame, sobel_kernel=15, thresh=(0.7, 1.3))
    
    thresh = (215, 255)
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    
    thresh = (90, 255)
    binary_2 = np.zeros_like(S)
    binary_2[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    merg = np.zeros_like(S)
    merg[(binary == 1) | (binary_2 == 1)]=1 #| (binary_v == 1 ) & (sobel_grad == 1 ) ]=1

    return merg 


def measure_curvature_pixels(warped_img: np.array, left_fit: np.ndarray, right_fit: np.ndarray):
    """

    measures the curvature of the road
    takes in a warped image and the left and right fits

    """

    y_coords = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    left_x_coords = left_fit[0]*y_coords**2 + left_fit[1]*y_coords + left_fit[2]
    right_x_coords = right_fit[0]*y_coords**2 + right_fit[1]*y_coords + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(y_coords)

    rescaled_y = y_coords*ym_per_pix
    rescaled_x_left = left_x_coords*xm_per_pix
    rescaled_x_right = right_x_coords*xm_per_pix
     
    left_fit_sc = np.polyfit(rescaled_y, rescaled_x_left, 2)
    right_fit_sc = np.polyfit(rescaled_y, rescaled_x_right, 2)

    left_curverad = ((1 + (2*left_fit_sc[0]*y_eval + left_fit_sc[1])**2)**1.5) / np.absolute(2*left_fit_sc[0])
    right_curverad = ((1 + (2*right_fit_sc[0]*y_eval + right_fit_sc[1])**2)**1.5) / np.absolute(2*right_fit_sc[0])
    
    return left_curverad, right_curverad

    
class CameraPipeline(object):

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny

        # hardcode perspective shift for now note relies on the 
        #self.persp_src = np.float32([[395, 180], [820, 180], [180,300], [1100,300]])
        #self.dest_src = np.float32([[190, 180], [1055, 180], [180,300], [1100,300]])

        self.persp_src = np.float32([[445, 150], [760, 150], [150,300], [1100,300]])
        self.dest_src = np.float32([[190, 150], [1055, 150], [150,300], [1100,300]])

        self.M = cv2.getPerspectiveTransform(self.persp_src, self.dest_src)
        self.Minv = cv2.getPerspectiveTransform(self.dest_src, self.persp_src)

        self.crop_image_margin = 0.05

        self.left_lane = None
        self.right_lane = None

    def calibrate_cam(self, cal_images: str):
        self.objpoints, self.imgpoints = genpoints(cal_images, self.nx, self.ny)

    def calc_polygon(self, img: str):
        output = adv_pipeline(img, self.objpoints, self.imgpoints)
        return output

    def _run_pipeline(self, frame: np.ndarray, crop_margin: float, transform: list, inv_transform: list, is_video: int):

        fr_shape = frame.shape

        # apply filter
        merg = filter_image(frame)

        # should have done in a mask?
        # simplistic selection of just the road section in front of the car
        car_forward_region = merg[int(fr_shape[0]/2):fr_shape[0],
                                        int(fr_shape[1]*crop_margin):int(fr_shape[1]*(1-crop_margin)) ]

        warped_shape = car_forward_region.shape 
        warped = cv2.warpPerspective(car_forward_region, transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)
        
        if is_video == 0:
            self.left_lane = None
            self.right_lane = None

        # returns the polynomial fit
        left_ft, right_ft, left_points, right_points, pointsy = fit_polynomial(warped, self.left_lane, self.right_lane)
        
        if is_video == 1:
            self.left_lane = left_ft
            self.right_lane = right_ft
        
        
        left_curverad, right_curverad = measure_curvature_pixels(warped, left_ft, right_ft)

        # drop the prior search if the curves go haywire
        if np.abs(left_curverad - right_curverad) > 1500:
            self.left_lane = None
            self.right_lane = None

        bias = calc_bias(warped, left_points, right_points)
        
        # plot the lines and section on warped colour section
        img_forward_region = frame[int(fr_shape[0]/2):fr_shape[0],
                                        int(fr_shape[1]*crop_margin):int(fr_shape[1]*(1-crop_margin)) ] 
        
        warp_img = cv2.warpPerspective(img_forward_region, transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)

        cv2_poly_points = plot_points(left_points, right_points, pointsy)

        image_set = np.zeros_like(warp_img)
        
        filled = cv2.fillPoly(image_set, np.int32([cv2_poly_points]),(10, 255, 0))

        un_warp = cv2.warpPerspective(filled, inv_transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)

        inv_warp_large = cv2.copyMakeBorder(un_warp, int(fr_shape[0]/2), 0, int(fr_shape[1]*crop_margin),
                                            int(fr_shape[1]*crop_margin), cv2.BORDER_CONSTANT, 0)

        return inv_warp_large, frame, bias, left_curverad, right_curverad

    def _run_video_pipeline(self, fil_path: str, crop_margin: float, transform: list, 
        inv_transform: list, is_video):

        frame, fr_shape = read_video(fil_path)

        inv_warp_large, frame, bias, left_curverad, right_curverad = self._run_pipeline(frame, crop_margin, transform, inv_transform, is_video)

        return inv_warp_large, frame, bias, left_curverad, right_curverad


    def process(self, img, is_video=1):

        """

        processes input with the pipeline
        if img is a string it opens it like a video path otherwise it expects an image array

        returns processed frames 

        """

        if type(img) is str:
            output, background, bias, left_curve, right_curve = self._run_video_pipeline(img, self.crop_image_margin, self.M, self.Minv, is_video=1)
        elif type(img) is np.ndarray:
            output, background, bias, left_curve, right_curve = self._run_pipeline(img, self.crop_image_margin, self.M, self.Minv, is_video=1)
            


        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        
        cv2.putText(background, 'bias (-ve is right drift): {0}m'.format(np.round(bias, 2)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale, fontColor, lineType )
        cv2.putText(background, 'left curve: {0}m'.format(np.round(left_curve, 2)), (50,75), cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale, fontColor, lineType )
        cv2.putText(background, 'right curve: {0}m'.format(np.round(right_curve, 2)), (50,100), cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale, fontColor, lineType )
        

        full_output = weighted_img(output, background) #, α=0.01, β=1)
        return full_output