# computer-vision
# Automatic detection of cardiomegaly in digital chest X-rays

# Table of contents 

Table of contents	1
Introduction	2
Rationale	3
Convolutional Neural Network	3
Conv2D	3
Pooling	3
Fully Connected Layer - Flatten and Output Layers	4
Models used for this study	4
Design	7
Loading and preprocessing our data	7
Image Data Generator	7
Flow From Directory	8
VGG-like baseline CNN architecture	9
Conv2D	9
Max Pooling	10
Fully connected layer	10
Flatten	10
Dropout	10
Dense	11
Compiling the model	11
Training the model	12
VGG16 with dropout	12
Inception V3 using transfer learning	13
Testing	14
VGG-like baseline CNN architecture	14
VGG16 with dropout	15
Inception V3 using transfer learning	16
Given those results, we can see that our model is underperforming compared to the previous one.	16
Conclusions	17
Python/Keras code used	18
Resources	18


# Introduction
The term "cardiomegaly" refers to an enlarged heart seen on any imaging test, including a chest X-ray. As a matter of fact, cardiomegaly is one of the most common inherited cardiovascular diseases with a prevalence at least 1 in 500 in the general population. It is a symptom of cardiac insufficiency which is a heart’s response to a variety of extrinsic and intrinsic stimuli that impose increased biomechanical stresses. While hypertrophy can eventually normalize wall tensions, it is associated with an unfavorable outcomes and threatens affected patients with sudden death or progression to overt heart failure.
 
In this report, we present an automated procedure to determine the presence of cardiomegaly on chest X-ray image based on deep learning.
The major objective of this study is to classify and recognize X-ray images with the presence of cardiomegaly compared to images with no findings. Overall, our data set consists of 2184 images: 1600 in the training set, 384 in the validation set and 200 in the test set. All of our datasets are splitted in two labels: 50% images with cardiomegaly and 50% images with no findings. 

Our analysis will show the test results on previously unseen images (test set 200 images) coming from three different convolutional neural network models: a VGG like architecture, a customized VGG16 and an InceptionV3 based fine-tuning based transfer learning architecture.

We will first explain the rationale behind our approach and then we will motivate the design choices made to build our models.
Finally, we will show and compare the performances of our models based on our test results and we will state our conclusions based on the key learnings and performance metrics.










# Rationale 
Convolutional Neural Network
To classify our dataset of X-ray images, we used convolutional neural networks also called CNNs. A Convolutional Neural Network is a Deep Learning algorithm, which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.
 
One of the key advantage of convolutional neural network is that the pre-processing required is much less than the one needed for other types of classification models. Moreover, a CNN during its training phase have the ability to capture the spatial and temporal dependencies of an image by automatically learning the relevant filters.


Our dataset is made of RGB input images and we have two different sizes (224X224 and 229X229) the first size will be used for our VGG-like model and the latter to feed the Inception V3 architecture.
The role of our convolutional neural network will be to reduce our input images into a form which is easier to process, without losing critical features needed to get a good prediction.
 
In the following sections, we will provide a general overview on the major architecture components used for the CNNs used in this study. 
 
## Conv2D
Convolution is needed to decrease the spatial size of our input images. The first convolution operation will take part in our Cov2D layer where a kernel K of dimension 3X3X1 will be involved. The objective of the Convolution Operation will be to extract the high-level features such as edges, from the input image. Conventionally, the first convolutional layer is responsible for capturing the Low-Level features such as edges, color, gradient orientation, etc. With added layers, the architecture adapts to the High-Level features as well as giving us a network which has the wholesome understanding of images in the dataset, similar to what the human brain does when processing images.

## Pooling
Similar to the Convolutional Layer, the Pooling layer is responsible for reducing the spatial size of the convolved feature by a predefined value. In our models, we will use a 2X2 matrix that will move across the input pixels and substitute a 2X2 area of the image with the maximum pixel value in the referenced area (in the case of max pooling). The rationale of this type of operation is to decrease the computational power required to process the data through dimensionality reduction and to prevent our models overfitting the training data. Furthermore, it is useful for extracting dominant features which are rotational and positional invariant, thus maintaining the process of effectively training of the model. For the purpose of our models, we have chosen Max Pooling which will returns the maximum value from the portion of the image covered by the Kernel.
The reason why we choose Max Pooling instead of average pooling is that the first one also performs as a Noise Suppressant. It discards the noisy activations altogether and also performs de-noising along with dimensionality reduction.
 
The Convolutional Layer and the Pooling Layer, together form the i-th layer of a Convolutional Neural Network. Depending on the complexities in the images, the number of such layers may be increased for capturing low-levels details even further, but at the cost of more computational power.
Fully Connected Layer - Flatten and Output Layers
Moving on with our architecture, we are going to flatten the final output into a column vector and feed it to a regular Neural Network for classification purposes. The also called densely connected layer will use the information present in the convolutional layers to decide which object is in the input image.

The flattened output is fed to a feed-forward neural network and backpropagation applied to every iteration of training.
Over a series of epochs, the model is able to distinguish between dominating and certain low-level features in images. Finally, since we are dealing with a binary classification problem, our model will be able to classify images using the Sigmoid classification technique.

Sigmoid function shape (From introduction to deep learning and convolutional neural network, 2017, Paul F Whelan)
Models used for this study
For the aim of this study, we applied three different convolutional neural network models. 
The first one is a VGG-like architecture while the second one is actually a copy of the VGG16 but with dropout layers integrated to its architecture. 
VGG is the model created by the visual geometry group at Oxford university which won the 2014 image net competition. 

Image source for VGG16: https://neurohive.io/en/popular-networks/vgg16/ 

VGG can be imported from Keras but the reason why we have created our custom one trained on CPU is that we have added a few Dropout layers.

Finally, our third model has been built using InceptionV3 leveraging transfer learning. With transfer learning, we will start with a neural network (InceptionV3 in this case) that has already been trained to recognize objects from a large data set like ImageNet. This has been done by slicing off the final layer of InceptionV3 and keeping all the layers that can detect patterns (basically our feature extractor). We will then create a new neural network that will substitute the last layer in the original network. Our training images will pass through the feature extractor and we will save our features for each image to a file.
Finally, we will use those extracted features to train our neural network. In this case, the last layer (our manually scripted NN) will be needed only to learn and to describe which pattern map to which cluster.


Image source for InceptionV3: https://cloud.google.com/tpu/docs/inception-v3-advanced 

## Design
In this section, we will describe in details the different design choices taken to build our three models. For the first two models, we used a similar design approach which can be declined in the following steps:
 
Loading data
Preprocessing data
Creating the CNN model
Compiling the model
Train the model
Evaluating the model
 
A different approach has been taken when using transfer learning, consequently our design process for the third model corresponds to the following steps:
 
## Loading data
Loading Keras InceptionV3 model to use as a feature extractor (freezing the top later)
Creating a new NN model including only the fully connected layer
Add the new NN to our InceptionV3
Perform model fine tuning on InceptionV3 by freezing some layers 
Preprocessing data
Compiling the model
Train the NN model using the extracted features from InceptionV3
Evaluating the model
 
In the following sections, we will explain more in details the structure of each of our model. However, since we used the same approach to load and preprocess our data, we will first go through this design decisions and then on the peculiarities of each model.

# Loading and preprocessing our data
## Image Data Generator
Prior to importing our data, we have created three variables containing the path to their respective folder for the training, test and validation set.
In order to import our data of X-ray images and to loop over our labelled folders, we used a pre-existent keras class called “ImageDataGenerator”.
This class generates batches of tensor image data with real-time data augmentation. Those batches have been created for our three data sets (training, test and validation) with data augmentation obviously applied only to our training set.
In terms of parameters for this class, we have mainly used its default values while normalizing the images by 1/255 making sure that our pixel values will be all included between 0 and 1 and in this way, we will allow the network to learn more quickly the optimal parameters for each input node. Finally as mentioned before, data augmentation has been applied only on our training dataset, in particular we have applied:
 
Horizontal flip
Width and height shift range of 10% of the total width/height
 
This decision has been made to train our network so to expect and handle off-center or horizontal images by artificially creating shifted and horizontal versions of the training data.

## Flow From Directory
In order to read set up the generators created in our previous step, we have used another keras class called “flow_from_directory”, which takes the path to a directory and passes the data to the generator which applies the specified augmentation. Here few crucial parameters have been specified:
Target size 224X224 for VGG and 229X229 for InceptionV3 (this parameter is not necessary since our images have been already resized in their respective folders)
Batch size → the size of batches of data that we want to create to train, validate and test our data is 20
Class mode is binary since our classification problem is binary (Cardio or no finding)
Color mode will be red green blue.
 
After taking those loading and preprocessing steps, we are ready to create our models, compile it and feed it with the right data structure. 



## VGG-like baseline CNN architecture 
In our first model, we have developed a basic VGG-like convolutional neural network architecture to classify our X-ray images. Below, we have included a picture of our model summary, which is composed by:


A classifier made of 3 convolutional layers and 3 pooling layers
A fully connected layer with 1 flatten layer, 1 dropout layer and 2 dense layers. 
 



# Conv2D
The convolutional layers used have all been set up in the same way. In particular for this model, we have used 32 output filters storing the results generated by a 3X3 convolution window. 
We used the variable “same” for padding, meaning that the input image ought to have zero padding so that the output in convolution will not differ in size compared to its input. In our first convolutional layer, we added the parameter input_shape to specify the shape of our input image (we used 224*224 to comply with the standard needed for the original VGG model). 
In order to avoid the vanishing gradient problem, we have decided to use the rectified linear unit (also called ReLu) activation function because it is not saturated and it does not squash the values into a small range as the other activations functions do ([0,1] or [−1,1]) rather it applies the max(0, x) thresholding at zero for every element. 

ReLu function (From introduction to deep learning and convolutional neural network, 2017, Paul F Whelan)
## Max Pooling 
The max pooling layers have been left with their default values meaning that we will use a 2X2 matrix that will move across the input pixels and substitute a 2X2 area of the image with the maximum pixel value in the referenced area. 
Fully connected layer 
## Flatten
For our flatten layer, we have used its default values. The reasons behind the use of this layer have been already explained in the previous section. 
## Dropout
For our dropout layer we have set our dropout rate to 20%, meaning that one in 5 inputs will be randomly excluded from each update cycle. The reason behind this choice depends on two factors: 
Our model has a moderate number of filters which will not impact dramatically its training speed 
Moreover, a dropout rate between 20% and 50% has been proven to be still efficient in terms of model accuracy and loss. 

## Dense 
Finally we have added two dense layers. Since we are dealing with a binary classification problem our last dense layer has size 1 and a sigmoid activation function (Sigmoid function: σ(x) = 1/1+e−x).  


# Compiling the model
Before training our model, we had to compile it and define three parameters:
The loss function - The loss function depends on the type of classification problem that we are trying to solve and since in our specific study we are dealing with a binary classification, binary cross entropy will be the loss function used (to add formula).
The optimizer function - To optimize the loss of our models, we have used the adaptive moment estimation (adam) optimizer. Adam is different to classical stochastic gradient descent. Stochastic gradient descent maintains a single learning rate (termed alpha) for all weight updates and the learning rate does not change during training. 
Adam optimizer use a similar approach to the so called  RMSProp and Adagrad optimizers with the difference that instead of adapting the parameter learning rates based on the average first moment (the mean), Adam also makes use of the average of the second moments of the gradients (the uncentered variance). 
Specifically, the algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages. As displayed in the following graph, adam optimizer is more efficient on deep learning models than others optimizers.

Chart source:  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ 

The type of metric – As for all of our models performances in this study will be measured on accuracy. Finally before training our model, we have computed the number of steps that should be run for each epoch as swell as the number of validation steps. This decision is based on the number of batches defined when creating our generators. This number will be 80 training steps and 19 validation steps (which is the result of samples over batches). 

# Training the model 
The number of epochs decided to train this model is 30. This choice depends on the relatively low number of filters (neural connections) used in this model as well as our computing resources. The training and validation performances will be discussed in our next section, what we can outline here is the actual training time which was equal to 6413 seconds or less than 2 hours.


# VGG16 with dropout 


The use of VGG16 has been done by manually replicating the original architecture including dropout layers (required for this assignment). The design decisions taken for data preprocessing are pretty much the same as the one taken for our previous model. Below is a summary of our model architecture:

The changes here compared to the original VGG16 model are related to the addition of 6 dropout layers (one placed at the end of every convolution/pooling combo with the last dropout layer placed in between the last two dense layers). This relatively aggressive use of dropout layers will prevent our model to overfit the training data and to increase the training speed (considering the high number of weights).
To compile this model, we have made a different design decision regarding the optimizer. This decision has been taken after running several trials using “adams” resulting in a high level of loss.
As a matter of fact, we have decided to use stochastic gradient descent with a low learning rate (0.0001) with decay, momentum and nesterov (add details here).
Finally, at the fitting stage, given our computational resources and given the complexity and deep of this architecture, we had to decrease the number of epochs to 15. Nevertheless, our design decision, our fit time was 80065 seconds or ~22 hours (20 hours more than our previous model).

# Inception V3 using transfer learning

In our last model, we used a similar approach for loading and preprocessing our data. The only difference is that to feed our inception V3 model, we had to change the scale of our image data from 224x224 (used before), to 229x229. Moreover, as outlined at the beginning of this section, in order to use transfer learning combined with image augmentation, we had to follow a different design process. 
After loading our datasets, we loaded the Inception V3 model (excluding its fully connected layer) that was pre-trained against the imageNet database and that will be used as a feature extractor. 
We then added a custom made densely connected classifier (Flatten, Dense 256, Dropout 0.5, Dense 1) on top of the inception base and we trained that model freezing all the weights trained on the imagenet classifier and using the same data augmentation and generators used in previous models. 

After that, to fine tune the feature extractor, we used a for loop that froze all the inception layers, leaving mixed 8 and mixed 9 still trainable. 
Finally, we compiled our model using “adam” (this time given the time constraint provided for this assignment we had to stick with this decision without trying other optimizers) and we trained the new fine tuned model using 15 epochs.


# Testing 
In this section, we will go through the models’ performances and testing results on previously unseen images. In particular, we will provide insights on each model accuracy on the test set as well as visualizing and commenting the results coming from the historical accuracy/loss results obtained during the training phase for each epochs, including a visual comparison between the training and validation accuracy.
 

Performance indicators
VGG16 like architecture
VGG 16 with dropout
InceptionV3
Accuracy
71%
50%
50%
Loss
0.59
0.69
~8
Predict time
50 secs
2258 secs
23 secs
Fit time
6413 secs
80000 secs
80065 secs

## VGG-like baseline CNN architecture 
This is the model where we obtained the best performance. The final accuracy score for this model computed on a test data of previously unseen images is 71% with an average loss equal to 0.59, predict time of ~50 seconds and fit time of 6413 seconds.
Our first line chart shows the changes in accuracy for each epoch on the training set (blue line) and on the validation set (yellow line). As we can see the two lines are following quite the same patterns meaning that the model is not overfitting and that there is an interesting generalization.
The second line chart represents the loss evolution over epochs comparing loss on the training set (blue line) as well as on the validation set (yellow line).


 


## VGG16 with dropout 
The final accuracy score on test data for this model is 50% with an average loss of 0.69, predict time of 2258 seconds and fit time of about 80000 seconds.
The accuracy chart (explained above) shows a different story compared to our previous model. In fact, the two lines are not following the same pattern and there is quite high variability in the values reported at different epochs (see for example validation line epoch 9 and 14). The same is happening for the loss chart, where we are seeing a decrease in loss (relatively low given our decision made on using a low learning rate) for the training line while a constant level around 0.69 for the validation line.
Given those results, we can see that our model is underperforming compared to the previous one. 

## Inception V3 using transfer learning 
The final accuracy score on test data for this model is 50% with a predict time of 23 seconds and a fit time of about 80065 seconds. Differently from the previous model, we decided to use here adam instead of the stochastic gradient descent (with a very low learning rate) loss function. 
The accuracy chart below shows a different story compared to our first model. In fact, the two lines are not following the same pattern and there is a quite high variability similar to the one seen in VGG16. Given the fact that our loss function is adam and not SGD with a low learning rate the loss values have a higher fluctuations compared to the previous model. Over epochs, the loss results do not seem to get better rather they stay constant, fluctuating from 7.8 to 8.2. 
Given those results, we can see that our model is underperforming compared to the previous one.


# Conclusions
VGG16 like architecture, VGG16 with dropout and a fine tuned InceptionV3 were the models deployed to conduct cardiomegaly detection. 
The accuracy of our models relied on previously labelled x-ray images divided in different folders depending on their respective scale. 

Our VGG16 like architecture is the model that performed well in recognising the disease and we were able to achieve a 71% in accuracy. The other two models (VGG16 and InceptionV3), given our design decisions as well as time constraints needed to try different hyperparameters, did not perform well: sometimes overfitting the training dataset and other times getting stuck into a local minimum (keeping the loss and accuracy values constant over epochs).

X-ray is only a 2D section of the 3D heart structure, whereas, in clinical practices, there are other advanced diagnostic methods such as Electrocardiogram, Echocardiogram, Cardiac computerized tomography or magnetic resonance imaging, which would provide extra information about how efficiently the heart is pumping and determine which chambers of the heart are enlarged. 
However, X-ray equipment is still the easiest access medical devices for screening cardiomegaly. As the accuracy and consistency of our solution increases the automatic diagnosis for cardiomegaly would have opportunities to replace manual screen drawing measurement and save millions of hours for radiologists in the future.


# Python/Keras code used
Load data
Os
Preprocessing
Image data generator
Flow from directory
Model Architecture
Sequential
Conv2D
MaxPoling
Flatten
Dense
Dropout
VGG16
InceptionV3
Layers
Models
Compile
Compile
Train
Fit generator
Test
Evaluate generator
Visualization/Performance
Time
Pyplot

# Resources
About cardiomegaly: https://www.mayoclinic.org/diseases-conditions/enlarged-heart/symptoms-causes/syc-20355436
About convolutional neural networks: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
About scarce model performances: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 
About adam optimizer: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ 
About InceptionV3: 
https://cloud.google.com/tpu/docs/inception-v3-advanced 
About VGG16: 
https://neurohive.io/en/popular-networks/vgg16/ 
Theoretical guide used: 
Introduction to deep learning and convolutional neural networks, 2017, Paul F Whelan
Why VGG was giving me high loss? I had to change the optimizer and learning rate: https://github.com/keras-team/keras/issues/7603
VGG like convnet with better optimizer and where to add droput layers: https://keras.io/getting-started/sequential-model-guide/ 
Which optimizer: https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/ 
Which parameter for flow from directory: https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720 
Why i used this dropout structure: https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5 


