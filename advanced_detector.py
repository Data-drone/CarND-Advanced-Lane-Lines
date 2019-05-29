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
def undistort_img(img: np.ndarray, objpoints: list, imgpoints: list) -> np.ndarray:
    """
    undistorts images based on object points

    """
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist,  rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# HSV color thresholder
def hls_select(img: np.ndarray, thresh: tuple=(0, 255)) -> np.ndarray:
    """
    receives an image as a numpy array applies a threshold on the s channel and outputs thresholded image

    """
    
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

def find_lane_pixels(binary_warped: np.ndarray)  -> Tuple[list, list, list, list, np.ndarray]:
    """

    applies the histrogram and sliding window method to find line pixels
    will return nones if cannot find

    """
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

def search_around_poly(binary_warped: np.ndarray, left_fit: list, 
    right_fit: list, margin: int=100) -> Tuple[list, list, list, list, np.ndarray]:

    """

    searches around a previous fit to find lane pixels

    """
    left_fit = left_fit[0]
    right_fit = right_fit[0]
    

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
    can return None for a fit if the input is none and lane pixels can't be found

    """
    
    # Find our lane pixels first from the image
    # haven't included code that will research if search fails yet
    if (left_fit is None or right_fit is None):
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty, out_img = search_around_poly(binary_warped, left_fit, right_fit)

    # Fit a second order polynomial to each using `np.polyfit`
    if leftx.size > 0:
        left_fi = np.polyfit(lefty, leftx, 2)
    else:
        left_fi = left_fit
    
    if rightx.size > 0:
        right_fi = np.polyfit(righty, rightx, 2)        
    else:
        right_fi = right_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fi[0]*ploty**2 + left_fi[1]*ploty + left_fi[2]
        right_fitx = right_fi[0]*ploty**2 + right_fi[1]*ploty + right_fi[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fi, right_fi, left_fitx, right_fitx, ploty


def calc_bias(img: np.ndarray, left_fit_pts: list, right_fit_pts: list) -> float:

    """

    calculates the bias, drift from center of the lane

    """
    
    middle = img.shape[1]/2

    lane_middle = (right_fit_pts[-1] - left_fit_pts[-1])/2 + left_fit_pts[-1]
    unscaled_bias = lane_middle - middle

    bias = ((unscaled_bias)*(3.7/800))

    return bias

def plot_points(left_fit_pts: list, right_fit_pts: list, ploty: list) -> list:
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


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)) -> np.ndarray:
    """
    
    directional threasholding function using sobel filters
    outputs a mask

    """
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


def color_filter(image: np.ndarray, lower: np.ndarray, upper: np.ndarray, colourspace = None) -> np.ndarray:

    if colourspace is not None:
        col_image = cv2.cvtColor(image, colourspace)
    else:
        col_image = image
    binary = cv2.inRange(col_image, lower, upper)

    return binary

def filter_image(frame: np.ndarray) -> np.ndarray:

    White_filter = color_filter(image = frame, lower = np.array([202,202,202]), upper = np.array([255,255,255]) )
    Yellow_filter = color_filter(image = frame, lower = np.array([200,120,0]), upper = np.array([255,255, 100]) )

    #Red_filter = color_filter(image = frame, lower = np.array([215, 0, 0]), upper = np.array([255, 0, 0]) )
    #HLS_filter = color_filter(image = frame, lower = np.array([0, 0, 90]), upper = np.array([0, 0, 255]), colourspace = cv2.COLOR_RGB2HLS )

    merged_binary = np.zeros_like(White_filter)
    merged_binary[(White_filter >= 1) | (Yellow_filter >= 1)]=1 #| (binary_v == 1 ) & (sobel_grad == 1 ) ]=1

    return merged_binary

def measure_line_curature(warped_img: np.array, line_fit: np.ndarray):

    """

    measures curvature of one single line

    """

    assert line_fit is not None

    y_coords = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    x_coords = line_fit[0]*y_coords**2 + line_fit[1]*y_coords + line_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 10/300 #30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    
    y_eval = np.max(y_coords)

    rescaled_y = y_coords*ym_per_pix
    rescaled_x = x_coords*xm_per_pix

    line_fit_sc = np.polyfit(rescaled_y, rescaled_x, 2)

    curve_rad = ((1 + (2*line_fit_sc[0]*y_eval + line_fit_sc[1])**2)**1.5) / np.absolute(2*line_fit_sc[0])
    
    return curve_rad
    
    

def measure_curvature_pixels(warped_img: np.array, left_fit: np.ndarray, right_fit: np.ndarray):
    """

    measures the curvature of the road
    takes in a warped image and the left and right fits

    """

    assert left_fit is not None
    assert right_fit is not None

    y_coords = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    left_x_coords = left_fit[0]*y_coords**2 + left_fit[1]*y_coords + left_fit[2]
    right_x_coords = right_fit[0]*y_coords**2 + right_fit[1]*y_coords + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 10/300 #30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    
    y_eval = np.max(y_coords)

    rescaled_y = y_coords*ym_per_pix
    rescaled_x_left = left_x_coords*xm_per_pix
    rescaled_x_right = right_x_coords*xm_per_pix
     
    left_fit_sc = np.polyfit(rescaled_y, rescaled_x_left, 2)
    right_fit_sc = np.polyfit(rescaled_y, rescaled_x_right, 2)

    left_curverad = ((1 + (2*left_fit_sc[0]*y_eval + left_fit_sc[1])**2)**1.5) / np.absolute(2*left_fit_sc[0])
    right_curverad = ((1 + (2*right_fit_sc[0]*y_eval + right_fit_sc[1])**2)**1.5) / np.absolute(2*right_fit_sc[0])
    
    return left_curverad, right_curverad


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        # just holds a y value to match best x
        self.besty = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = []  
        #y values for detected line pixels
        self.ally = []

    def check_fit(self):
        """

        function that checks the fit to make sure we return something sensical
        we need to return a:

            bestx / besty / radius_of_curvature needed att the end

        """

        # no current fit
        if self.current_fit[0].size == 1:
            pass
        
        else:
            self.best_fit = self.current_fit
            self.bestx = self.allx[-1]
            self.besty = self.ally[-1]

            # untested
            self.detected = True


    def add_fit(self, x: list, y: list, warped: np.ndarray):
        """

        take a set of x and y from fits
        check that there are values
        add it to all x and all y
        run a current fit
        
        

        """  

        if (len(x) > 0) and (len(y) > 0):
            self.allx.append(x)
            self.ally.append(y)

            line_fit = np.polyfit(y,x,2)
            if len(line_fit) > 0:
                self.current_fit = [line_fit]
                self.radius_of_curvature = measure_line_curature(warped, line_fit)
        
        else:
            self.current_fit = [np.array([False])]  
        
        self.check_fit()

            


    
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

        self.left_line = Line()
        self.right_line = Line()
        

    def calibrate_cam(self, cal_images: str):
        """

        calibrates for camera distortion based on checkerboard samples

        """
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

        # warps the image to get the birds eye view
        warped = cv2.warpPerspective(car_forward_region, transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)
        
        """
        look at both lines, is the detected flag set
        if no then run the fit
        """

        if (self.left_line.detected == False) or (self.right_line.detected == False):
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        else:
            leftx, lefty, rightx, righty, out_img = search_around_poly(warped, 
                                                                        self.left_line.best_fit,
                                                                        self.right_line.best_fit)
            
        self.left_line.add_fit(leftx, lefty, warped)
        self.right_line.add_fit(rightx, righty, warped)            

        # use the best fit from each to create x and y for plots

        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fit_pts = self.left_line.best_fit[0][0]*ploty**2 + self.left_line.best_fit[0][1]*ploty + self.left_line.best_fit[0][2]
        right_fit_pts = self.right_line.best_fit[0][0]*ploty**2 + self.right_line.best_fit[0][1]*ploty + self.right_line.best_fit[0][2]


        """
        # make sure we always calculate if not in video mode
        if is_video == 0:
            self.left_lane = None
            self.right_lane = None

        # returns the polynomial fit - can return zero if it can't find anything
        left_ft, right_ft, left_points, right_points, pointsy = fit_polynomial(warped, self.left_lane, self.right_lane)
        
        if is_video == 1:
            if left_ft is not None:
                self.left_lane = left_ft
            if right_ft is not None:
                self.right_lane = right_ft
            if left_points is not None:
                self.left_x = left_points
                self.left_y = pointsy
            if right_points is not None:    
                self.right_x = right_points
                self.right_y = pointsy
        
        # calculates lane curvature
        left_curverad, right_curverad = measure_curvature_pixels(warped, self.left_lane, self.right_lane)

        # drop the prior search if the curves go haywire
        if np.abs(left_curverad - right_curverad) > 1500:
            self.left_lane = None
            self.right_lane = None

        """

        bias = calc_bias(warped, left_fit_pts, right_fit_pts)
        
        # outputs:
        # 
        

        # plot the lines and section on warped colour section
        # inputs are: initial frame
        # crop margin
        # left_points, right_points
        # 
        img_forward_region = frame[int(fr_shape[0]/2):fr_shape[0],
                                        int(fr_shape[1]*crop_margin):int(fr_shape[1]*(1-crop_margin)) ] 
        
        warp_img = cv2.warpPerspective(img_forward_region, transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)

        cv2_poly_points = plot_points(left_fit_pts, right_fit_pts, ploty)

        image_set = np.zeros_like(warp_img)
        
        filled = cv2.fillPoly(image_set, np.int32([cv2_poly_points]),(10, 255, 0))

        un_warp = cv2.warpPerspective(filled, inv_transform, (warped_shape[1], warped_shape[0]), flags=cv2.INTER_LINEAR)

        inv_warp_large = cv2.copyMakeBorder(un_warp, int(fr_shape[0]/2), 0, int(fr_shape[1]*crop_margin),
                                            int(fr_shape[1]*crop_margin), cv2.BORDER_CONSTANT, 0)

        return inv_warp_large, frame, bias, self.left_line.radius_of_curvature, self.right_line.radius_of_curvature

    def _run_video_pipeline(self, fil_path: str, crop_margin: float, transform: list, 
        inv_transform: list, is_video):
        """

        uses the cv2 based read_video function instead to open videos 

        """

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