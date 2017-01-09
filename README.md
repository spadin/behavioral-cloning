# Behavioral Cloning Project

## Table of Contents

1. [Model architecture design](#model-architecture-design)
1. [Architecture characteristics](#architecture-characteristics)
1. [Model training](#model-training)

## Model architecture design

A model with the following architecture is used.

```python
# crops incoming image, removes sky and car hood
Lambda(crop, input_shape=(160, 320, 3), name="crop")

# normalizes input values to between -1.0 and 1.0
Lambda(normalize, name="normalize")

# 2x2x3 convolution with 32 filters, stride of 2
Convolution2D(32, 2, 2, subsample=(2, 2), border_mode="same")
# activation
LeakyReLU()

# 2x2x3 convolution with 64 filters, stride of 2
Convolution2D(64, 2, 2, subsample=(2, 2), border_mode="same"))
# activation
LeakyReLU()

# pooling layers, pool_size of 2
MaxPooling2D()

# 2x2x3 convolution with 2 filters, stride of 2
Convolution2D(2, 2, 2, subsample=(2, 2), border_mode="same"))
# activation
LeakyReLU()

# 2x2x3 convolution with 4 filters, stride of 2
Convolution2D(4, 2, 2, subsample=(2, 2), border_mode="same"))
# activation
LeakyReLU()

# pooling layers, pool_size of 2
MaxPooling2D()

# flatten data for fully-connected layers
Flatten()

# fully connected layer
Dense(100)
# activation
LeakyReLU()

# fully connected layer
Dense(50))
# activation
LeakyReLU()

Dense(1)
```

## Architecture characteristics

I started building my model as a single layer deep neural network. I wanted to
build the simplest network possible before I started adding convolutions and
layers. I worked on the pipeline and generating images from the drive log first
in order to make sure the model was getting good data before expanding on the
model.

Once I felt the pipeline was in a good place, I added a convolution layer to
the model. After tuning the number of filters and the convolution size several
times without a successful run around the track, I decided to add another
convolution layer and a max pooling layer. I still had not completed a full
circuit around the track, so I doubled the convolution layers and added
another max pooling layer.

At that point, with the model above, I was able to complete a run around the
track, however, I was crossing some yellow lines of going on the curbs. I
wanted to keep the size of the convolutions small so I could have multiple
convolutions, so I decided to tune the number of filters per convolution. After
several iterations of experimentation I landed on some hyperparameters I was
happy with.

For calculating loss, I used mean-squared error. I didn't find the MSE to be
directly correlated to best performance. My best trained model, for example,
had a validation loss of 0.0066, but the run with the lowest validation loss
was 0.0052, and it crossed the yellow lines. I used the MSE to guide me towards
which runs I should try run in the simulator first, but that's about it.

This model is only suitable for this track, it doesn't generalize well. To
generalize the model more, I could randomly change the brightness of some of
the training images so the model can train in different lighting scenarios. I
would also have to train the model in the second track so it can train on
different sloped roads.

## Model training

The model was trained on an AWS g2.2xlarge EC2 instance. The model ran for a
total of 100 epochs. At each epoch, the checkpoint of the model would be
created, which saved the weights at that epoch.

The left, right and center images are used in training. I've added 0.1 (2.5
degrees) to the left images and -0.1 to the right images to compensate for the
difference in camera angle.

Here's an example of the three different types of images.

**Left image**

![Left image](./images/left_2016_12_19_20_10_35_002.jpg?raw=true)

**Center image**

![Center image](./images/center_2016_12_19_20_10_35_002.jpg?raw=true)

**Right image**

![Right image](./images/right_2016_12_19_20_10_35_002.jpg?raw=true)
