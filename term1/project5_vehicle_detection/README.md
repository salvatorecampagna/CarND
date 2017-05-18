# Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_car]: ./output_images/car1.jpg
[image_noncar]: ./output_images/noncar1.jpg
[image_car_hog]: ./output_images/car_hog.png
[image_noncar_hog]: ./output_images/noncar_hog.png
[image_test1]: ./output_imaes/test1.jpg
[image_test2]: ./output_images/test2.jpg
[image_test3]: ./output_images/test3.jpg
[image_test4]: ./output_images/test4.jpg
[image_test5]: ./output_images/test5.jpg
[image_test_all_boxes]: ./output/test_all_boxes.jpg
[image_pipeline1]: ./output_images/pipeline1.png
[image_pipeline2]: ./output_images/pipeline2.png
[image_pipeline3]: ./output_images/pipeline3.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---

The full code is included in the Jupyter Notebook located at `./vehicle_detection.ipynb`.

[Video](./project_video_vehicles.mp4) - Vehicle detection video.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cell 3 of the Jupyter notebook in function **get_hog_features**.  

I started creating a dataset including `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image_car]
![alt text][image_noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a `vehcile`:

![alt text][image_car_hog]

and for a `non-vehicle':

![alt_text][image_noncar_hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and a decided the combination to use only after checking the performance of the classifier.
The set of parameters I decided to use in the end is available in cell 8.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in cell 12. The classifier training is performed using 14208 images including both `vehicle` and `non-vehicle` images. Then, 3553 images including `vehicle` and `non-vehicle` images, are used to test the classifier performances on unseed data.
At the end the classifier has an accuracy of more than 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In cell 16 the **find_cars** function is available which is used to select a set of windows at different positions and different scales.
I use five different scales (1.0, 1.5, 1.75, 2, 2.25) and search the original image limiting the region of interest to the area delimited by y values in the range [380, 700]. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

Here are some examples:

![alt text][image_test1]
![alt_text][image_test2]
![alt text][image_test3]
![alt_text][image_test4]
![alt text][image_test5]
![alt_text][image_test_all_boxes]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_vehicles.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In cell 23 functions `add_heat()` and `apply_threshold()` are provided, which are used to create a heat map and threshold the heat map to identify vehicles in the image. In the same cell `draw_labeled_bboxes()` uses `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map. Then, each one of these blobs is assumed to correspond to a vehicle which is then bounded by a box by the same functions.

Here's an example result showing the heatmap from a series of frames of video:

![alt_text][image_pipeline1]
![alt_text][image_pipeline2]
![alt_text][image_pipeline3]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline has mainly two problems which need improvement:
1. It is slow at processing the video, while I would expect that a real world application needs real-time performances in order for vehicle detection to be done in a real sel driving car
2. It has still some false positive detctions even if all of them are present for a very short time

So possible improvements improvements include making the processing faster and improving false positive vehicle detections filtering. Possible ways to improve the pipeline include:
1. Using a Convolutional Neural Network to do vehicle detection instead of a Support Vector Machine classifier
2. Using a bigger dataset possibly applying data augmentation to improve the performances of the classifier
3. Find cars using more coarse grained windows at first so to be able to search the next frame of the video only on a limited area.
4. Tracking objects so to predict their position and search the next frame of the video only in the area where the object is predicted to be 
