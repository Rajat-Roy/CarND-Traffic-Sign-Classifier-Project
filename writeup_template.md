# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup-images/histogram.jpg "Visualization"
[image2]: ./writeup-images/unprocessed-with-labels.jpg "Un-Processed"
[image3]: ./writeup-images/processed.jpg "processed"
[image4]: ./writeup-images/unlabeled.jpg "un labeled"
[image5]: ./writeup-images/predict.jpg "predicted"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Rajat-Roy/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:
```
Number of training examples = 34799
Number of validation examples = 32
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There are two steps in the image pre-processing step:
 * equalize the exposure by applying histogram equalization technique
 * normalize the resulting image
We can clearly see in the input images that the lighing condition are different.

![alt text][image2]

 So, we apply the above method and get better and even outputs.
 The difference between the original data set and the augmented data set is the following ... 
 
![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x90 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x90 				|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 9x9x60	|
| RELU					|	
| Max pooling	      	| 2x2 stride,  outputs 4x4x60 				|
| Flatten | 4x4x60 = 960 |
| Fully connected		| outputs 900        									|
| RELU |
| Fully connected		| outputs 512        									|
| RELU |
| Fully connected		| outputs 256        									|
| RELU |
| Fully connected		| outputs 128        									|
| RELU |
| Fully connected		| outputs 43        									|
| RELU |
| Softmax				| outputs 43         |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I adapted from the LeNet architecture with few modifications.
* Depending on the number of classification, which is 43 in this case, I chose to set 90 filters for the first conv-net, a filter size of 7x7 gave good results.
* For the second conv-net, I chose 5x5 filter to shrink the output at reduced the filters to 60
* Then after flattening the outputs from the conv-net I was left with 960 features.
To classify these features into 43 classes I used 5 Fully Connected neural networks.
It gradually classified them into half the number of features each time finally getting 43 classes.

The model was trained for maximum of 50 epochs. But it is also conditioned to stop early if it reaches 98% accuracy on the validation set.

A batch size of 210 and learning rate of 0.0011 produced better results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.981
* test set accuracy of 0.965

```
Final evaluations are:
Training set Accuracy = 1.000
Validation set Accuracy = 0.981
Test set Accuracy = 0.965
```
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![alt text][image4]
These are un-labeled.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction with predicted labels:
![alt text][image5]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.


The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9935523         			| Speed limit (30km/h)   									| 
| 1.0     				| Yield 										|
| 0.9929087					| Road narrows on the right											|
| 1.0	      			| Road	work				 				|
| 1.0				    | Keep right      							|

```
Top sofmax probability indices:
[ 1 13 24 25 38]

Top sofmax probability values:
[0.9935523 1.        0.9929087 1.        1.       ]

Predicted Labels:
[ 1 13 24 25 38]

Actual Labels:
[1, 13, 24, 25, 38]

Correct Predictions:
[ True  True  True  True  True]
```
