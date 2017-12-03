# Semantic Segmentation Project
Self-Driving Car Engineer Nanodegree Program


[sample_13]: ./images/umm_000013.png "Sample 13"
[sample_15]: ./images/umm_000015.png "Sample 15"
[sample_29]: ./images/umm_000029.png "Sample 29"
[sample_39]: ./images/umm_000038.png "Sample 39"
[sample_49]: ./images/umm_000049.png "Sample 49"
[sample_63]: ./images/umm_000063.png "Sample 63"


## Background

In this project a **Fully Convolutional Network (FCN)** is used for **Semantic Segmentation**. Semantic Segmentation is the task of partitioning an image into semantically meaningful parts, and to classify each part into one of the pre-determined classes. The FCN used in this project
can partition images identifying pixels belonging to the road or not. As a result the network
receives images of any size as input and produces images of the same size as output identifying
pixels representing roads.

## FCN Architecture

By the architectural point of view a Fully Convolutional Network is comprised of two parts: an
`encoder` and a `decoder`. The encoder is made by a series of convolutional layers whose goal is to extract features from an image. The decoder up-scales the output of the encoder in such a way to
get an output image whose size is the same of the input image. In this project a pre-trained
VGG-16 neural network is used as the decoder. VGG-16 is a deep neural network for image
classification which normally does not suit the purpose of semantic segmentation.
For this reason it has been modified to remove the upper fully connected layers
and preserving lower convolutional layers and their weights. This way the encoder preserves its
feature extraction capabilities.
The output of the last layer of VGG-16 is taken as input to the decoder part. At this point `1x1
convolutions` are used to replace the fully connected layers and preserve spatial information in
the image. Following 1x1 convolutions is a sequence of `transposed convolutions` whose purpose is
to up-scale the image to the original size.
To further improve the semantic segmentation performance `skip connections` are included allowing
the network to use information from multiple resolution scales.

## FCN Optimizer

The optimizer used to train (fine tune) the network is the `Adam optimizer`, which is optimizing the `cross-entropy loss` function to classify each pixel as road or not road.

## FCN Training

To train the FCN network the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) is used from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  
Training this type of network is quite time consuming and demanding in terms of computational
resources. As a result the training has taken place on a machine with a GPU.  

The hyperparameters used for training the FCN are the following:

* Epochs (`epochs`): 50
* Keep probability (`keep_prob`): 0.5. Training keep probability used for dropout (to prevent overfitting). During testing the value is set t 1.0 in such a way to use the full network
capability.
* Batch size (`batch_size`): 5 images per batch of training.
* Learning rate (`learning_rate`): 0.00005.

After 10 epochs the cross-entropy loss is in the range `[0.08 - 0.05]`. After 30 epochs is in the range `[0.03 - 0.02]` while at the end of epoch 50 it is in the range `[0.02 - 0.01]`.


## Sample images

A list of some sample images follws as they are produced by the FCN, with the segmentation class overlaid on top of the original image in green.

![alt_text][sample_13]
![alt_text][sample_15]
![alt_text][sample_29]
![alt_text][sample_39]
![alt_text][sample_49]
![alt_text][sample_63]
