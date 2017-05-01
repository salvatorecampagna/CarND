*Behavioral Cloning* 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_dataset_distrib]: ./images/dataset_distrib.png "Dataset steering angle distribution"
[image_trainset_distrib]: ./images/trainset_distrib.png "Dataset steering angle distribution (training set)"
[image_validset_distrib]: ./images/validset_distrib.png "Dataset steering angle distribution (validation set)"
[image_testset_distrib]: ./images/testset_distrib.png "Dataset steering angle distribution (test set)"
[image_center_track1]: ./images/center_2016_12_01_13_31_14_702.jpg "Image center driving"
[image_recovery1]: ./images/recovery/center_2016_12_01_13_42_28_197.jpg "Recovery Image (1)"
[image_recovery2]: ./images/recovery/center_2016_12_01_13_42_28_298.jpg "Recovery Image (2)"
[image_recovery3]: ./images/recovery/center_2016_12_01_13_42_28_400.jpg "Recovery Image (3)"
[image_recovery4]: ./images/recovery/center_2016_12_01_13_42_28_502.jpg "Recovery Image (4)"
[image_recovery5]: ./images/recovery/center_2016_12_01_13_42_28_604.jpg "Recovery Image (5)"
[image_recovery6]: ./images/recovery/center_2016_12_01_13_42_28_705.jpg "Recovery Image (6)"
[image_recovery7]: ./images/recovery/center_2016_12_01_13_42_28_806.jpg "Recovery Image (7)"
[image_recovery8]: ./images/recovery/center_2016_12_01_13_42_28_906.jpg "Recovery Image (8)"
[image_recovery9]: ./images/recovery/center_2016_12_01_13_42_29_007.jpg "Recovery Image (9)"
[image_recovery10]: ./images/recovery/center_2016_12_01_13_42_29_109.jpg "Recovery Image (10)"
[image_epochs_overtraining]: ./images/overtraining_epochs.png "Training epochs (overtraining)"
[image_epochs]: ./images/training_epochs.png "Training epochs"


**Rubric Points**
---

***1. Submission includes all required files and can be used to run the simulator in autonomous mode***

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode on track ona and two (10 mph)
* drive1.py for driving the car in autonomous mode on track one (25 mph)
* drive2.py for driving the car in autonomous mode on track two (12 mph)
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* run1.mp4 track one video capture
* run2.mp4 track two video capture 

***2. Submission includes functional code***

Using the Udacity provided simulator and drive.py, drive1.py and drive2.py files, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
```sh
python drive1.py model.h5
```
```sh
python drive2.py model.h5
```

***3. Submission code is usable and readable***

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

***1. An appropriate model architecture has been employed***

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 76 - 97). 

The model includes ReLU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 82).

**2. Attempts to reduce overfitting in the model**

To reduce overfitting I included a single dropout layer, just before the output layer (model.py ine 95). 

The model was trained and validated on different data sets, splitting the original dataset in training, validation and test set.
The original dataset includes 108555 images, including images coming from center, left and right camera. 70% of these images make the training set (75988), 12% the validation set (13027) and 18% the test set (19540).
The distribution of the steering angle for the whole dataset and for the training, validation and test datasets is shown in the following figures.

![alt_text][image_dataset_distrib]
![alt_text][image_trainset_distrib]
![alt_text][image_validset_distrib]
![alt_text][image_testset_distrib]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on both the first and the second track.

***3. Model parameter tuning***

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 173).

***4. Appropriate training data***

Training data was chosen to keep the vehicle driving on the road. I captured data combining center lane driving, recovering from the left and right sides of the road. I also collected data driving on both track one and two in the two directions.

For details about how I created the training dataset, see the next section. 

**Model Architecture and Training Strategy**

***1. Solution Design Approach***

The architecture of the model is inspired by the Nvidia architecture suggested in the lesson.
I thought this model might be appropriate as a first attempt.

In order to gauge how well the model was working, I split my image and steering angle data into a training, validation and test set. I started training my network for a large number of epochs, 20, just to see after how many epochs it was overfitting.
Looking at the validation set mean squared error I could find the point where it started to increase after decreasing and I chose that point as the maximum value for the training epochs, which is 10. I also used a test set to confirm the result after completing training and tuning all parameters. 

To combat overfitting I also added a dropout layer just before the output layer.

I also tried two different loss functions, Mean Squared Error (MSE) and Mean Absolute Error (MAE), finding MSE providing better results when driving the car in the simulator. I also tried different optimizers including Adam, RMSProp, SGD and Adagrad. The final choise was to use Adam since it was converging fater to the optimal solution. 

The final step was to run the simulator to see how well the car was driving around track one and two. At the first attempts there were a few spots where the vehicle fell off the track. The car was also oversteering to the left and had problems steering to the right. To improve the driving behavior in these cases I did the following:
* Augment the datset including more samples of the part of the track which had problems.
* Collect more training samples by driving on the track in the opposite direction. This has the effect of balancing the dataset for left and right turns.

At the end of the process, the vehicle is able to drive autonomously around both the tracks without leaving the road.

***2. Final Model Architecture***

The final model architecture (model.py lines 76-97) consisted of a convolution neural network with the following layers and layer sizes:
* Lambda layer: used to normalize each input image
* Cropping layer: to crop the input image excluding 70 lines at the top and 25 lines at the bottom
* Convolutional layer: filter size 5x5, strides 2x2, depth 24 and ReLU activation
* Convolutional layer: filter size 5x5, strides 2x2, depth 36 and ReLU activation
* Convolutional layer: filter size 5x5, strides 2x2, depth 48 and ReLU activation
* Convolutional layer: filter size 3x3, depth 64 and ReLU activation
* Convolutional layer: filter size 3x3, depth 64 and ReLU activation
* Flattening layer
* Dense layer: size 128
* Dense layer: size 64
* Dense later: size 32
* Dropout layer: used to reduce overfitting (keep_probability: 0.5)
* Dense layer: size 1, which outputs the steering angle

***3. Creation of the Training Set & Training Process***

To capture good driving behavior, I first recorded a couple of laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_center_track1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust the direction when driving off the center. These images show what a recovery looks like starting from left and adjusting to the center:

![alt text][image_recovery1]
![alt text][image_recovery2]
![alt text][image_recovery3]
![alt text][image_recovery4]
![alt text][image_recovery5]
![alt text][image_recovery6]
![alt text][image_recovery7]
![alt text][image_recovery8]
![alt text][image_recovery9]
![alt text][image_recovery10]

Then I collected data on track two repeating the same process in order to get more data points.

To augment the data set, I also drove the car in the opposite direction on both track one and two. This was done both to improve the robustness of the model and also to balance left and right turns into the dataset. Balancing left and right turns helps preventing oversteering in one direction.

After the collection process, I had 36185 samples, each sample including an image coming from the center camera and two images coming from the left and right camera for a total of 10855 images. To derive the steering angle for the left and right camera images, I applied a correction to the original steering angle adding 0.1 to the steering angle for the left camera and subtracting 0.1 to the steering angle for the right camera.

I finally randomly shuffled the data set and put 70% of the data into the training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the first figure below, since the validation starts to increase afterwards. After determining the right number of epochs I re-trained the network for only 10 epochs as shown in the second figure below.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image_epochs_overtraining]
![alt_text][image_epochs]

After that I tested the model by driving the car in autonomous mode on both the first and second track.

I also wanted to test my model to see if it was generalizing well to different driving conditions, so I did the following:
* I tried my model also on another track using the old version of the simulator. This simulator includes an additional track which the model has never seen before and could be used to check if the model can drive the car on a track that is new to it. My model can *almost* drive on this additional track making some minor mistakes on a couple of turns (hitting the border in a couple of points).
* Increase the driving speed making the model more robust. I succesfully tested my model to drive at higher speed on both track one and two, being able to drive at 25 mph on track one and 12 mph on track two.