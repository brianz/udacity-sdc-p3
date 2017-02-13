# Udacity Self Driving Car Nanondegree 

## Project 3: **Behavioral Cloning** 

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Code layout

This project consists of the following files:

- This `README.md` file which you're reading
- `model.py` is the single Python file which implements the entire CNN model. Output after running
  this file is a single model file with the current timestamp embedded in the name.
- `drive.py` which is used to connect the simulator to the trained model and steer the car based
- `model.h5` model file which, due to it's size, is stored on S3: http://brianz-udacity-sdc.s3.amazonaws.com/p3/model.h5

## Training data

For this project I used the Udacity-provided data set along with my own manually collected data sets.

The driving data I collected can be broken into two sections:

- "Normal" driving around the track
- "Hard turns" and critical turns along the track

### Normal driving

The normal driving which I performed was simply driving the track either in the center of the road or on the right 
side of the road. I originally had another data set where I drove on the left side of the road, however I found with
that data set performance was much worse where the car would drive too far to the left and off the road. By removing
the left side driving the results were much better

### Hard turns and critical turns

Initially, I started driving in automomous mode with just the "normal driving" data. I quickly found that the car had
no idea how to handle driving near or off the side of the road and would never recover. To deal with this, I 
created datasets where I would perform the following at different parts of the track:

- while starting off driving on the right or left lane, start recording and turn hard to get to the center of the road
- while headed off/pointing towards the side of the road, start recording and turn hard to re-center
- at critical parts of the track, namely big turns, make successive recordsing ensuring to make hard rather
  than gradual turns

#### Hard turns after driving on lane
![Hard left turn from right lane](https://github.com/brianz/udacity-sdc-p3/blob/master/hard-left-turn-while-driving-on-lane.gif)

![Hard right turn from left lane](https://github.com/brianz/udacity-sdc-p3/blob/master/hard-right-turn-while-driving-on-lane.gif)

#### Hard turn to center after heading off road

![Hard turn back after heading off side of road](https://github.com/brianz/udacity-sdc-p3/blob/master/hard-left-turn-after-heading-to-right-lane.gif)

![Hard turn to center after heading off road near big rock](https://github.com/brianz/udacity-sdc-p3/blob/master/hard-right-turn-after-big-rock-apex.gif)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


## Usage

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
curl -O http://brianz-udacity-sdc.s3.amazonaws.com/p3/model.h5
python drive.py model.h5
```

## Model Architecture and Training Strategy

`model.py` includes an implemenation of the Nvida model which can be found at https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/.
This model can be seen in the [`get_model` function](https://github.com/brianz/udacity-sdc-p3/blob/master/model.py#L120)
and consists of:

- normalization
- 3 `5x5` convolutions with 3, 24 and 36 output layers, each with a `2x2` stride and a Relu activation
- 2 `3x3` convolutions with 48 and 48 output layers, each with a `1x1` stride and Relu activation
- 1 flattening layer
- 4 fully connected layers with 1164, 100, 50 and 10 output layers, each with Relu activation
- 1 finaly fully connected layer with a single output which consists of the final steering angle prediction


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
