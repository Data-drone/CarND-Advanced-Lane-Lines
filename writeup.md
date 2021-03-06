## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_test_1]: ./test_images/test1.jpg "Before Undistory"
[image_undist_test_1]: ./output_images/undist_test1.jpg "Before Undistory"
[image_filter_test_1]: ./output_images/filteredtest1.jpg "After Filter"
[image_prewarp]: ./output_images/prewarp.jpg "Before Warp"
[image_postwarp]: ./output_images/postwarp.jpg "After Warp"
[image_done]: ./output_images/final_test1.jpg "final output"
[final]: ./project_processed_video.mp4 "Video"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to compute the camera matrix and distortion coefficients is run in `advanced_detector.py` There is a `calibrate_cam` function as part of the `CameraPipeline` class. This runs the `genpoints` function which is in lines 10 to 31. It is run in the Calibration Image section of the IPython notebook `Build Advanced Detector.ipynb`

The output images were created using `calibrate_demo.py`. genpoints works by taking an image in a preset (x,y) grid then using `cv2.findChessboardCorners` to output a tuple containing a list of object points and image points respectively. These are used by the `cv2.CalibrateCamera` in the `undistort_img` function. 

#### 1. Provide an example of a distortion-corrected image.

Raw Image
![alt text][image_test_1]

Undistorted Image
![alt text][image_undist_test_1]


### Pipeline (single images)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After the undistort step, I take the red channel of the image and after a HLS conversion, the S channel and threshold them to try and extract the lines more clearly. For R channel, I use threshold 215 to 255
and for S channel I use 90 to 255. These are then merged together. 

The filtering code is in lines 243 to 256 in the `advanced_detector.py` file. See the Filter cell of `Build_Advanced_Detector.ipynb` to see the end result. (unable to save out image for some reason)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is hardcoded with surface and destination points as follows:

```python
persp_src = np.float32(
    [[445, 150], 
     [760, 150], 
     [150,300], 
     [1100,300]]
    )
dest_src = np.float32(
    [[190, 150], 
     [1055, 150], 
     [150,300], 
     [1100,300]]
    )

```

This is run through the `cv2.getPerspectiveTransform` function with the transformation matrix being used in a Linear Intepolation via `cv2.warpPerspective` with a cropped version of the whole frame. The crop was chosen to focus on just the road surface to minimise clutter for the lane pixel detection step.

The test of the perspective transform is in the workbook `Build_Advanced_Detector.ipynb` with an example being


PreWarp Image
![alt text][image_prewarp]

PostWarp Image
![alt text][image_postwarp]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


To identify the lines, I fit a polynormal to the line pixels identified. This process is defined in the `find_lane_pixels` in `advanced_detector.py` file. The lane pixel identifier uses 9 sliding windows looks for non-zero pixels within each window.

A second degree polynomial is fitted to the identified pixels via the `fit_polynomial` function in `advanced_detector.py`. For video scenarios, where there can be priors, there is a check at `line 472` that determines whether the `find_lane_pixels` function is used or the `search_around_poly` function which just runs a search around the location of the line identified in the previous frame.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the `measure_line_curvature` function in the `advanced_detector.py`. It uses the functions as explained in Measuring Curvature I video.

The position of the vehicle with respect to the center, the bias, is calculated in the `calc_bias` function. This is calculated by looking at the distance of the left line from the middle of the image, the distance from the right line and hence working out the offset.

The factors for converting pixels to meters is based on analysing the postwarp image which suggested that for my case, the lane is about 800 pixels width and the look ahead region is roughly three road line dashes or 10 meters ahead.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 398 through 407 in my code in `advanced_detector.py` in the function `process_image()`.  See the Run on Test cell of `Build_Advanced_Detector.ipynb` to see the end result. (unable to save out image for some reason)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_processed_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Sense checking lines is a bit of a trial and error scenario with the limits say on road curvature hard coded. it would make sense for this to be more dynamic to handle a wider variety of cases. At the moment, the pipeline reverts to previous fit if it cannot find a good new fit. There is currently no mechanism to retire the previous fit if it gets too old, adding this kind of functionality will help to make the algorithm more robust.

In the advanced video, there are scenes where one line is missing. Ideally the algorithm should have a concept of how far road lines are apart and be able to through a combination of using prior knowledge and knowing rough lane widths be able to work out where the other line would be should only one line appear in the video.

In cases where there are no painted road lines and road edges are where pedestrian pavement or wild bushland starts, the road line filters are likely to fail and hence a lane will not be detected. Addressing this will require new filter logic, perhaps leveraging edge detectors rather than just the colour detectors currently used.


The perspective transform has been hardcoded rather than being dynamic so in areas when the perspective changes due to undulating curves in the road, the warp won't result in a good candidate for lane detection.