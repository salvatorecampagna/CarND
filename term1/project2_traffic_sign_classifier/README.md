**Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[barchart]: ./examples/barchart.png "Traffic Sign frequency"
[no_vehicle_grayscale]: ./examples/no_vehicle_grayscale.png "No Vehicle grayscale"
[test_image1]: ./test_images/keep_right.png "Keep right"
[test_image2]: ./test_images/no_passing_over_35.png "No passing over 3.5"
[test_image3]: ./test_images/road_work.png "Road work"
[test_image4]: ./test_images/slippery_road.png "Slippery road"
[test_image5]: ./test_images/speed_limit_30.png "Speed limit (30km/h)"
[test_image6]: ./test_images/speed_limit_70.png "Speed limit (70km/h)"

## Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
---
This file describes the project and all the steps carried out to implement a Traffic Sign classifier using a Convolutional Neural Network implemented in TensorFlow.
The project itself is provided as a Python Jupyter Notebook available at the following link: [project code](https://github.com/salvatorecampagna/CarND/blob/master/term1/project2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb)

The Jupyter Notebook is exported also as an HTML file and is available at the following link: [project html](https://github.com/salvatorecampagna/CarND/blob/master/term1/project2_traffic_sign_classifier/Traffic_Sign_Classifier.html)

## Data Set Summary & Exploration

***Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.***

The Python code providing basic summary of the data is contained in the 3rd code cell of the Jupyter notebook. I reported the shapes of the datasets, in such a way to understand how datasets are organized and how many samples are available for training, validation and testing of the classifier.

Summary statistics of the traffic signs data set:

+ The size of the training set is 34799
+ The size of the test set is 12630
+ The shape of a traffic sign image is 32 x 32 pixels
+ The number of unique classes/labels in the data set is 43

I used Pandas, at this stage, to load the table mapping traffic sign IDs to traffic sign labels. This way I have a table mapping, for instance, traffic sign ID 2 to label 'Speed limit (50km/h)' and so on.

***Include an exploratory visualization of the dataset and identify where the code is in your code file.***

In the 4th cell I plot 10 random sample images from the training set.

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency of each traffic sign in the training dataset.

![alt text][barchart]

In the 5th cell I plot a frequency table and an histogram showing the frequency for each traffic signs in the training set, ordering them by their frequency. As we can see there are traffic signs whose frequency is higher, such as 'Speed limit (50km/h)', 'Yield' or 'Keep right' and traffic signs which are less frequent, such as 'Go straight or left', 'Dangerous curve to the left' or 'Speed limit (20km/h)'.

For this reason particular care is required when evaluating the performance of the classifier since not all classes are equally represented in the training set.

## Design and Test a Model Architecture

***Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.***

The code for this step is contained in the 6th, 7th and 8th code cell of the Jupyter notebook.

As a first step, I decided to convert images from RGB to grayscale. Apparently colors do not convey useful information for classification. Neurons in different layers detect features like shapes, lines and contrast which are not related to the color itself of the traffic sign. Moreover, when switching from RGB to grayscale images the accuracy of the classifier was not affected.
As a result, feeding the network with RGB images, instead of grayscale images, does not bring any improvement to the classification accuracy. Using grayscale images instead of RGB images, instead, simplifies and speeds up the network training.

As a last step, I normalized the images using Min-Max scaling so to have values in the range [0.1, 0.9]. This way gradient descent and similar optimization algorithms converge much faster to the optimal solution.

Here is an example of a traffic sign image after grayscaling.

![alt text][no_vehicle_grayscale]

***Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)***

Training, validation and test datasets are already provided. I didn't need to further split them. I just shuffled the training set before training the network, so to be sure to pick random batches at each epoch.

I used the training set for training the network and the validation set to tune the hyperparameters (more details later). Once done, I used the test set to test the classifier on previously unseen data.

The final training set has 34799 samples, the validation set has 4410 samples, while the test set has 12630 samples. All images are grayscale images of size 32 x 32 pixels.

I didn't augment the dataset since I decided I would have done dataset augmentation as the last thing to try if performances were not good enough.
This because of the following reasons:
+ Augmenting the dataset takes computing time to actually manipulate images (rotate, flip, increase contrast and/or brightness, add additional sources of light,...)
+ Augmenting the dataset increases training and validation time of the network
+ Augmenting the dataset increases the memory required to store additional images

I also guessed that, due to different traffic sign frequencies, it made more sense, eventually, to do a dataset augmentation after observing the actual performances of the classifier in classifying each traffic sign class. Being a multiclass classification problem, I guessed performances of the classifier would have changed depending on the traffic sign class. For this reason, applying data augmentation to each and every class does not make much sense to me. I would rather approach the problem of augmenting the training set just for traffic sign classes whose performances are not good enough. This way the effort required to augment the dataset is minimal and the size of the training set is kept at minimum too.

I decided to start with an architecture that resembles the popular LeNet-5 architecture and just observe its performance on traffic signs classification. After evaluating the performances of LeNet-5 I procedeed to improving it.
As a matter of fact LeNet-5 already has an overall accuracy of about 90% 'as is'.

### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10th cell of the Jupyter notebook.

My final model consisted of the following layers:

| Layer         		| Description       	        					|
|:---------------------:|:-------------------------------------------------:|
| Input         		| 32x32x1 grayscale image       					|
| Convolution 3x3       | 1x1 stride, valid padding, output 30x30x6 	    |
|                       | parameters: 60 (54 weights, 6 biases)             |
| ReLU activation		|							    					|
| Convolution 5x5       | 1x1 stride, valid padding, output 26x26x16        |
|                       | parameters: 2416 (2400 weights, 16 biases)        |
| ReLU activation		|								    				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 22x22x32      	|
|                       | parameters: 12832 (12800 weights, 32 biases)      |
| ReLU activation		|						                            |
| Max Pooling 2x2       | 2x2 stride, valid pooling, output 11x11x32        |
| Flattening layer      | input: 11x11x32, output: 3872                     |
| Fully connected		| input: 3872, output: 120                          |
|                       | parameters: 464760 (464640 weights and 120 biases)|
| Dropout               | Keep probability: 0.6 (training)                  |
| Fully connected       | input: 120, output: 84                            |
|                       | parameters: 10164 (10080 weights and 84 biases)   |
| Output			    | input: 84, output: 43                             |
| Softmax               | 43 one-hot-encoded vectors                        |
|:---------------------:|:-------------------------------------------------:|
|                       | Total parameters: 493887                          |
|                       | (493586 weights and 301 biases)                   |


### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 12th cell of the Jupyter notebook.

To train the model, I used the AdamOptimizer. The hyperparameters used during training are the following:
+ Epochs: 100
+ Batch size: 160
+ Learning rate: 0.001
+ Train dropout keep probability: 0.5

The set of hyperparameters has been selected by manually and iteratively tuning each paremeter using the validation set.

Input image data in the training set has been normalized so to improve convergence of the optimization algorithm.

### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Training of the network takes place in the 12th cell. In the same cell, after each epoch the validation set is used to
evaluate the performance of the network after each epoch. For each epoch the loss and accuracy is computed on both the training set and test set. Comparing the loss and accuracy on the training set and validation set allows to check for overfitting.
Moreover, two plots are displayed in the 13th cell, showing the training and validation loss and accuracy as a function of the number of training epochs. Cell 14 reports the result on the test set.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 98%
* test set accuracy more than 96%

I chose and iterative approach:
+ I started with a network resembling the architecture of LeNet-5, which had an accuracy of about 90%.
+ I first tried to normalize data using Min-Max scaling to improve the AdamOptimizer convergence to the optimal solution.
+ I then decided to try using grayscale images instead of RGB images to reduce the amount of data to process and speedup training time. I found that having RGB images does not really matter since the accuracy doesn't change.
+ At this point I used the validation set to find a good set of hyperparameter values including the optimizez learning rate, the batch size and the number of epochs.
+ After that, I tried to increase the number of neurons in differet layers making the network wider. This was not providing meaningful improvements so I discarded this option.
+ Then, I tried removing the two Max Pooling layers at the output of the two first convolutional layers and replaced them with dropout. I added dropout between the two fully connected layers. Using dropout makes the network more robust against overfitting since, the network, cannot rely on the presence of a single neuron but rather have more neurons sharing the effort of learning a specific feature. After adding the dropout layer I had to use an iterative approch to find a good value for the additional hyperparameter, the keep probability of the dropout layer. Once found a good value for the keep probability I had a better accuracy score, around 93%, and I decided to keep dropout in place of the two Max Pooling layers.
+ Now, since a wider network was not providing better results, I tried to make the network deeper adding an additional convolutional layer. I wanted to try adding a small layer so to avoid adding too many parameters. This way I avoid increasing too much the training time of the network. For this reason I added a 3x3 convolutional layer with ReLU activation just after the input layer and before the first convolutional layer. That gave some improvement on the accuracy.
+ While doing all these iterations I have always plotted two graphs showing the loss and accuracy, on the training and validation sets, as functions of the number of epochs to make sure the network was not underfitting or overfitting.
+ Summarizing, I kept a LeNet-5 based network arcitecture with one additional 3x3 convolutional layer after the input layer, removed the two Max Pooling layers at the output of the second and third convolutional layer and added dropout between the first and the second fully connected layers. With this network architecture the accuracy on the test set is between 96% and 97%.
+ After 100 epochs the accuracy on the training set is around 99%, on the validation set is around 98%, while the accuracy on the test set is more than 96%. These values are quite close to each other so the network is not underfitting or overfitting.

In cell 15 and 16 I also analyzed the performance of the classifier plotting a confusion matrix and computing the precision, recall and F1 score of the classifier for each of the 43 different output classes. The confusion matrix shows that the classifier has a different classification accuracy for each one of the 43 different traffic sign classes. This means, for instance that the classifier is better at classifying traffic signs such as 'Dangerous curve to the left' and 'No entry' than it is in classifying traffic signs such as 'End of no passing' or 'Pedestrians'. This information could be axploited, for instance, if considering the idea of doing data augmentation of the training set to improve the performances of the classifier. A strategy could be to augment the dataset just for traffic signs whose classifier accuracy is low, such as 'End of no passing' or 'Pedestrians'.

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][test_image1] ![alt text][test_image2] ![alt text][test_image3]
![alt text][test_image4] ![alt text][test_image5] ![alt text][test_image6]

I chose the traffic sign 'Speed limit (30km/h)' since it is easy to mistake it for 'Speed limit (80km/h)', while I chose the traffic sign 'No passing for vehicles over 3.5 metric tons' since it is easy to mistake it for 'End of no passing by vehicles over 3.5 metric tons'. For the others there is no specific reason.

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th cell of the Jupyter notebook.

Here are the results of the prediction:

| Image			        						|     Prediction	        					|
|:---------------------------------------------:|:---------------------------------------------:|
| Keep right   									| Keep right									|
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons	|
| Road work 									| Road work 									|
| Slippery Road									| Slippery Road      							|
| Speed limit (30km/h)							| Speed limit (30km/h)							|
| Speed limit (70km/h)							| Speed limit (70km/h)							|

The model was able to correctly guess 6 of the 6 traffic signs, therefore its accuracy is 100%. 
This compares favorably to the accuracy on the test set which is more than 96%.

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model and report softmax probabilities is located in the 21st cell of the Jupyter notebook.

For the first image, the model is relatively sure about the prediction. It predicts correctly the traffic sign as a 'Keep right' traffic sign with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| Keep right   									    |
| ~0.00    				| Turn left ahead                                   |
| ~0.00					| Speed limit (20km/h)							    |
| ~0.00				    | Speed limit (30km/h)   						    |
| ~0.00				    | Speed limit (50km/h) 							    |


For the second image, the model is relatively sure about the prediction. It predicts correctly the traffic sign as a 'No passing for vehicles over 3.5 metric tons' traffic sign with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| No passing for vehicles over 3.5 metric tons  	|
| ~0.00    				| Speed limit (80km/h)                              |
| ~0.00 				| End of no passing by vehicles over 3.5 metric tons|
| ~0.00	      			| Vehicles over 3.5 metric tons prohibited          |
| ~0.00				    | No passing            						    |

For the third image, the model predicts correctly the traffic sign as 'Road work' with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| Road work   									    |
| ~0.00    				| Keep right                                        |
| ~0.00					| Bicycles crossing									|
| ~0.00	      			| Dangerous curve to the right					    |
| ~0.00				    | Go straight or right      						|

For the fourth image, the model is again quite sure about the prediction. It predicts correctly the traffic sign as a 'Slippery road' traffic sign with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| Slippery road     							    |
| ~0.00    				| Bicycles crossing                                 |
| ~0.00					| Dangerous curve to the right					    |
| ~0.00	      			| No passing					 				    |
| ~0.00				    | Children crossing   						        |

For the fifth image, the model is quite sure about the prediction. It predicts correctly the traffic sign as a 'Speed limit (30km/h)' traffic sign with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| Speed limit (30km/h)							    |
| ~0.00    				| End of speed limit (80km/h)                       |
| ~0.00					| Speed limit (20km/h)							    |
| ~0.00	      			| Speed limit (70km/h)							    |
| ~0.00				    | Speed limit (80km/h)							    |

For the sixth image, the model is sure about the prediction. It predicts correctly the traffic sign as a 'Speed limit (70km/h)' traffic sign with a probability very close to 1.00. The top five soft max probabilities were:

| Probability         	|     Prediction	        					    |
|:---------------------:|:-------------------------------------------------:|
| ~1.00        			| Speed limit (70km/h)							    |
| ~0.00    				| Speed limit (20km/h)							    |
| ~0.00					| Speed limit (30km/h)							    |
| ~0.00	      			| Stop      					 				    |
| ~0.00				    | General caution       						    |