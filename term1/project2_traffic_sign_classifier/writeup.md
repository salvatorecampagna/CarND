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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
---
This file serves the purpose of describing the project and all the steps carried out to implement a Traffic Sign classifier.
The project itself is provided as a Python Jupyter Notebook available at the following link: [project code](CarND/term1/project2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb)

The Jupyter Notebook is exported also as an HTML file and is available at the following link: [project html](CarND/term1/project2_traffic_sign_classifier/Traffic_Sign_Classifier.html)

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The Python code providing basic summary of the data is contained in the third and fourth code cells of the Jupyter notebook. I first reported the shapes of the datasets, in such a way to understand how datasets are organized and how much data is available for training, validation and testing of the classifier.

I also used Pandas, at this stage, to load the table mapping traffic sign ids to traffic sign labels. This way I have a table mapping, for instance, traffic sign id 2 to label 'Speed limit (50km/h)' and so on.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 pixels
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The Python code for this step is contained in the fifth code cell of the Jupyter notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency of each traffic sign in the training dataset.

![alt text][barchart]

In the fifth cell I also show a frequency table. The frequency table provides frequency data for each traffic sign, ordering them by their frequency. As we can see there are traffic signs whose frequency is higher, such as 'Speed limit (50km/h)', 'Yield' or 'Keep right' and traffic signs which are less frequent, such as 'Go straight or left', 'Dangerous curve to the left' or 'Speed limit (20km/h)'.

For this reason particular care is required when evaluating the performance of the classifier.

## Design and Test a Model Architecture

### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth, seventh and eighth code cell of the Jupyter notebook.

As a first step, I decided to convert the images from RGB to grayscale. Apparently colors do not convey useful information for classification. Neurons in different layers detect features like shapes, lines and contrast which are not related to the color itself of the traffic sign.

Feeding the network with RGB images, instead of grayscale images, does not bring any visible improvement to the classification accuracy. Using grayscale images instead of RGB images also simplifies and speeds up the network training.

As a last step, I normalized the image data using Min-Max scaling in such a way that values range in the interval [0.1, 0.9]. This way gradient descent and similar optimization algorithms converge much faster to the optimal solution.

Here is an example of a traffic sign image after grayscaling.

![alt text][no_vehicle_grayscale]

### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data is already split since the beginning in training, validation and test set so I didn't need to further split it. I used the provided training set, validation set and test set as they are. I just shuffled the training set before training the network to be sure to pick random batches at each epoch.

My final training set has 34799 grayscale samples, each being 32 x 32 pixels. My validation set has 4410 grayscale samples, same shape as the training set, while the test set has 12630 images with same shape as the training set images.

I didn't augment the dataset since the classifier has already an overall accuracy between 96% and 97%. Moreover, I didn't want to do dataset augmentation nefore having a look at the performance of the classifier. My guess was that, due to different traffic sign frequencies, it makes more sense, eventually, to do a dataset augmentation driven by the performance of the classifier in classifying each class.

### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the Jupyter notebook.

My final model consisted of the following layers:

| Layer         		| Description   	        					                |
|:-----------------:|:-------------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							          |
| Convolution 3x3   | 1x1 stride, valid padding, output 30x30x6 	      |
|                   | parameters: 60 (54 weights, 6 biases)             |
| ReLU activation		|												                            |
| Convolution 5x5   | 1x1 stride, valid padding, output 26x26x16        |
|                   | parameters: 2416 (2400 weights, 16 biases)        |
| ReLU activation		|												                            |
| Convolution 5x5	  | 1x1 stride, valid padding, output 22x22x32      	|
|                   | parameters: 12832 (12800 weights, 32 biases)      |
| ReLU activation		|												                            |
| Max Pooling 2x2   | 2x2 stride, valid pooling, output 11x11x32        |
| Flattening layer  | input: 11x11x32, output: 3872                     |
| Fully connected		| input: 3872, output: 120                          |
|                   | parameters: 464760 (464640 weights and 120 biases)|
| Dropout           | Keep probability: 0.6 (training)                  |
| Fully connected   | input: 120, output: 84                            |
|                   | parameters: 10164 (10080 weights and 84 biases)   |
| Output				    | input: 84, output: 43                             |
| Softmax           | 43 one-hot-encoded vectors                        |
|:-----------------:|:-------------------------------------------------:|
|                   | Total parameters: 493887                          |
|                   | (493586 weights and 301 biases)                   |


### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 12th cell of the Jupyter notebook.

To train the model, I used the AdamOptimizer. The hyperparameters used during training are the following:
+ Epochs: 100
+ Batch size: 160
+ Learning rate: 0.001
+ Train dropout keep probability: 0.6

Input image data in the training set has been normalized so to improve convergence of the optimization algorithm. I have tried mutiple different combinations
of the parameters before coming up with this set.

### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
