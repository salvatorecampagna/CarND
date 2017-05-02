# Advanced Lane Finding Project

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

[chess_undist]: ./output_images/undistorted_chessboard.jpg "Undistorted chessboard"
[image_undist]: ./output_images/test_undistorted.jpg "Undistorted image"
[test_gradx]: ./output_images/test_gradx.jpg "Gradient along X axis"
[test_r]: ./output_images/test_r.jpg "Red channel thresholding"
[test_warped]: ./output_images/test_warped.jpg "Perspective transform"
[test_warplane]: ./output_images/test_warplane.jpg "Lane lines on warped image"
[test_lane]: ./output_images/test_lane.jpg "Lane lines on undistorted image"
[test_curv_dist]: ./output_images/test_curv_dist.jpg "Curvature and vehicle position"
[test_full_pipeline]: ./output_images/pipeline.png "Full pipeline"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

The full code is included in the Jupyter Notebook located at `./advanced_lane_detection.ipynb`.

[First video](./project_video_lanes.mp4) - Processed by slower pipeline

[Second video](./project_video_lanes_fast.mp4) - Processed by faster pipeline

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is provided in cell 3 by function `calibrate_camera()` and is used in cell 4.

The `calibrate_camera()` function starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time. I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][chess_undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Image undistortion is provided in cell 6 by function `image_undistort()`. This function is used to undistort input images coming from the camera and needs the camera matrix and undistortion coefficients computer by `calibrate_camera()` above. Here is an example:
![alt text][image_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code for all different thresholding techniques used is provided in cell 9. All these functions are then combined in cell 16 by function `image_thresholds()`. Here's an example of gradient along X axis and red color thresholding:

![alt text][test_gradx]
![alt text][test_r]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for performing perspective transform is provided in cell 12 by function `image_warp()`. This function takes as input an image (`image`) and  uses source (`src`) and destination (`dst`) points to transform the image from the camera view to a bird's eye view.  I chose to hardcode the source and destination points, which are the following:

```python
src = np.float32([[575, 460],
                  [715, 460],
                  [1150, 720],
                  [240, 720]])
dst = np.float32([[440, 0],
                  [950,0],
                  [950,720],
                  [440, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 440, 0        | 
| 715, 460      | 950, 720      |
| 1150, 720     | 950, 720      |
| 240, 720      | 440, 0        |

I verified that my perspective transform was working as expected by drawing a test image and its warped counterpart in cell 13 to verify that the lines appear parallel in the warped image.
Here is a sample warped image:

![alt text][test_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

Lane line pixels are identified on the warped image by function `slide_window_searc()` provided in cell 17. It returns fit coefficients for a second order polynomial which are then used by function `image_lane()` to display the lane lines on the original undistorted image and on the warped image:

![alt text][test_warplane]
![alt text][test_lane]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Computing radius of curvature of the lane and the position of the vehicle with respect to the center is performed by function `compute_curvature_and_distance()` provided in cell 21.

I also verified in the video that the radius of curvature in the first curve is about 1 KM.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Cell 23 provides the function `image_curv_dist()` which plots curvature and vechicle position on the image in such a way to have the curvature and vechicle position in the final video.
Here is an example:

![alt text][test_curv_dist]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a sample input frame as processed by the full pipeline:

![alt text][test_full_pipeline]

Here's a [link to my video result](./project_video_lanes.mp4)

---

### Discussion


#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my first attempt I tried using just gradient along X and Y, gradient magnitude and gradient direction to identify lane lines. After trying on some test images I realized that this was not good enough, so I explored different possibilities such as using also color thresholding. I have found that using the 'b' channel of the Lab colorspace is quite effective in identifying yellow lines while the 'S' channel of the HLS colorspace is good to identify white lines. So I combined the following thresholding techniques:
* Gradient along X
* Gradient along Y
* Gradient magnitude along both X and Y
* Gradient direction
* HSL S channel
* Lab b channel

At this point my pipeline was working fine on test images and also on the first part of the video, but it failed with lane lines crossing in the final part of the video. For this reason I collected some more images from the video (images available in directory `./other_test_images`) and tested my pipeline against those 'difficult' images. The pipeline output (after improvement) on these images is available in cell 25 of the Jupyter Notebook.

I realized that my pipeline was not good enough to identify lane lines in images with shadows and different light conditions. As a result, I had to fine tune the thresholds trying different values for both the gradient and color thresholds. I also replaced the HLS S thresholding with the Red RGB thresholding.

At the end I combined the following thresholding techniques:
* Gradient along X
* Gradient along Y
* Gradient magnitude along both X and Y
* Gradient direction
* RGB R channel
* Lab b channel

This pipeline can still fail on roads with shadows or different light conditions, so possible improvements could be:
* using different color spaces and color thresholding techniques to better identify lane lines
* replace some of the pipeline stages with different techniques to improve the pipeline performance and get a pipeline which could process images in real-time
* using information such as the width of a lane to improve lane line detection. For instance, it should be possible to find one of the lane lines with high confidence and then compute the other line by just computing a parallel line whose distance is equal to the lane line width. For instance, identify the left lane line and then compute the right lane line by fitting a line that is 'width' pixels on the right with respect to the left lane line