# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
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
[image_pipeline1]: ./output_images/pipeline1.png
[image_pipeline2]: ./output_images/pipeline2.png
[image_pipeline3]: ./output_images/pipeline3.png
[image_pipeline4]: ./output_images/pipeline4.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---

The full code is included in the Jupyter Notebook located at `./vehicle_detection.ipynb`.

[Video](./project_video_output.mp4) - Vehicle detection video.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cell 3 of the Jupyter notebook in function **get_hog_features**.

I started creating a dataset including `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image_car]
![alt text][image_noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a `vehcile`:

![alt text][image_car_hog]

and for a `non-vehicle`:

![alt_text][image_noncar_hog]

At the end I used the following parameters for extracting the HOG features:
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and a decided the combination to use only after checking the performance of the classifier.
At the same time I also tried to keep the number of features low when selecting the features to use to avoid problems with the dimensionality of the dataset.
The set of parameters I decided to use in the end is available in cell 9.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training the classifier features are scaled in cell 11 using a `StandardScaler()` and the dataset is split in **train** and **test** set.
I trained a linear SVM in cell 13. The classifier training is performed using 15271 images including both `vehicle` and `non-vehicle` images. Then, 3818 images, including `vehicle` and `non-vehicle` images, are used to test the classifier performances on unseed data.
At the end the classifier has an accuracy of more than 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In cell 16 the **find_cars** function is available which is used to select a set of windows at different positions and different scales.
It computes binned color features and color histogram features on those different windows while computes hog features only once for the whole image. After that, it feeds the classifier and gets the classifier output. If the classifier identifies the box as containing a car the box is added to a list of `hot_windows`.
I preferred using the **find_cars** function instead of a pure sliding window approch since find_cars is faster as a result of computing hog features for the whole image.
Going back to the choice of parameters for feature extraction I tried different combinations and I found that having more features did not provide significant improvements to the performance of the classifier. Also I didn't want to unnecessarly increase the dimentionality of the dataset.

I realized also doing multiple experiments that the classifier is not very effective with scales lower than 1.50 and higher than 2.00. For this reason I limited searching for windows only using 1.50 and 2.00 scales. Moreover, I searched the original image limiting the region of interest to the area delimited by y values in the range [360, 680]. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result.

Here are some examples showing the full pipeline:

![alt_text][image_pipeline1]
![alt_text][image_pipeline2]
![alt_text][image_pipeline3]
![alt_text][image_pipeline4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In cell 18 functions `add_heat()`, `heatmap_threshold()` and `heatmap()` are provided, which are used to create a heat map and apply  a threshold to the heat map to identify vehicles in the image and remove false positives. In the same cell `draw_labeled_bboxes()` uses `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map. Then, each one of these blobs is assumed to correspond to a vehicle which is then bounded by a box by the same function and displayed on the original image.
Moreover, in the pipeline function available in cell 20 a list of heat maps is kept and a the fuction `heatmap_threshold()` is used to apply a threshold to the heat map which is obtained by summing all the heat maps in the list.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced the following problems while working on this project:
1. Making the pipeline fast enough to keep the processing in the 'it/s' range. At some point I realized that the classifier was making a lot of mistakes (false positives) when using scales lower than 1.50 and higher than 2.00. So I decided to use a ROI made by only two 'sub-ROIs' and searched those ROIs for scales of 1.50 and 2.00 only. As a result of this the pipeline processes images faster and also false positive detections of the classifier are reduced.
2. The other problem was reducing the number of false positives. To address this I did the following things:
* I augmented the dataset including both `vehicle` and `non-vehicle` images. I used images from the Udacity dataset and from the project video itself to add `vehicles`. To add `non-vehicles`, instead, and reduce the number of false positives, I also captured some images and false positive detections from the project video
* I used the LinearSVC `decision_function()` in **find_cars** to discard car detections too close to the decision boundary. This could result in some correct car detections to be rejected but has the effect of reducing the number of false positives. The pipeline is robust enough, anyway, to tolerate correct detected cars rejected by the additional check done using `decision_function()`
* I used a kind of `heat map averaging` keeping a list of the most recent heat maps and then thresholding on the sum of all the heat maps in the list. This provides robustness against false positives and has also the effect of smoothing the boxes around detected cars


Possible improvements for the vehicle detection pipeline are:
1. Improving the performance of the pipeline so to run in real-time
2. Finding cars using more coarse grained windows at first so to be able to search the next frame of the video only on limited areas
3. Tracking objects so to predict their position and search the next frame of the video only in the area where the object is predicted to be (considering that anyway a new car can apper in the frame)
