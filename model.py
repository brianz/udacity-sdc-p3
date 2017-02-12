# coding: utf-8

import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from datetime import datetime
from keras.layers import Activation, Dropout, Flatten, Dense, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, Sequential
from pprint import pprint as pp
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


Record = namedtuple('Record', 'center, left, right, angle')


def normalize_fpath(directory, path):
    path, fn = os.path.split(path.strip())
    return os.path.join(directory, 'IMG', fn)


def take_record(angle, directory):
    # When looking at data which I recorded manually, take everything. Just use a simple naming
    # convention for all of my directories.
    if directory.startswith('bz'):
        return True

    # When the angle is non-zero we'll always take it
    if angle != 0.0:
        return True

    # Take 20% of the records which have a 0.0 steering angle
    if np.random.uniform() < 0.20:
        return True

    return False


def build_img_lists(directory):
    # columns are:
    #
    #  center,left,right,steering,throttle,brake,speed
    root = os.path.split(directory)[0]
    fn = os.path.join(directory, 'driving_log.csv')

    if not os.path.isfile(fn):
        print('Skipping %s due to missing driving_log.csv' % (directory, ))
        return []

    print('Processing files in', directory)

    with open(fn) as fh:
        reader = csv.reader(fh)
        lines = (
            (
                normalize_fpath(directory, l[0]),
                normalize_fpath(directory, l[1]),
                normalize_fpath(directory, l[2]),
                float(l[3]),
            ) for l in reader if take_record(float(l[3]), directory)
        )
        return [r for r in map(Record._make, lines)]


def crop_img(path):
    """crop the top 60 pixels out"""
    return misc.imread(path)[60:]


def _data_generator(X, batch_size):
    """Infinite generate which returns a given amount of data.

    When the generator reaches the end of the data stream it will roll over so that the data stream
    is essentially a ring which never runs out.

    Here, note we are **only** using the center images, not the left or right images.
    """
    start = 0
    end = batch_size
    size = len(X)
    while True:
        data = X[start:end]
        # check if we're at the end of the stream and need to pad extra
        if len(data) < batch_size:
            extra_end = batch_size - len(data)
            extras = X[0:extra_end]
            data.extend(extras)

        # Pull out just the center images and corresponding angles
        records = [(r.center, r.angle) for r in data]
        images = np.asarray([crop_img(r[0]) for r in records], dtype="float32")
        angles = [r[1] for r in records]

        yield (images, angles)

        start += batch_size
        end += batch_size
        if start > size:
            start = 0
            end = batch_size


def training_generator():
    """Generator for training data"""
    return _data_generator(training_set, batch_size=256)


def validation_generator():
    """Generator for validation data"""
    return _data_generator(training_set, batch_size=64)


def get_model(input_shape):
    """Creat a CNN based on the Nvidia model documented here:

    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

    Note, there are no Dropout layers for a few reasons:

    - Testing showed actual results were **worse** with Dropout layers
    - Empirical evidence showed results improved more with a specific type of test data rather than
      adding Dropout
    - We do trim out a big amount of zero degree test data when buliding up our list of training
      data
    - This is (hopefully) and exact duplicate of the Nvidia CNN which didn't use Dropout

    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape, name="Lambda1"))
    model.add(Convolution2D(3, 5, 5, subsample=(2, 2), activation='relu', name="Conv3",
                            border_mode='valid'))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu', name="Conv24",
                           border_mode='valid'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu', name="Conv36",
                           border_mode='valid'))
    model.add(Convolution2D(48, 3, 3, activation='relu', name="Conv48",
                           border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation='relu', name="Conv64",
                           border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu', name="Dense1164"))
    model.add(Dense(100, activation='relu', name="Dense100"))
    model.add(Dense(50, activation='relu', name="Dense50"))
    model.add(Dense(10, activation='relu', name="Dense10"))
    model.add(Dense(1))
    model.summary()
    return model


# Build up a list of all the data and split/shuffle for training and validation
records = []
for directory in glob.glob('training-data/*'):
    recs = build_img_lists(directory)
    records.extend(recs)

training_set, validation_set = train_test_split(
    records,
    test_size=0.2,
    random_state=0,
)

print(len(training_set))
print(len(validation_set))

# create a single image to figure out our global shape
img = crop_img(training_set[0].center)
img_shape = img.shape
print('Using image shape:', img_shape)

# Sanity check...spit out a plot to see how many zero degree steering angles/images we have
_angles = np.asarray([r.angle for r in training_set])
print(np.min(_angles))
print(np.max(_angles))
plt.hist(_angles, bins=100, color= 'orange')
plt.xlabel('steering value')
plt.ylabel('counts')
plt.show()

# total number of angles
print("Total number of steering angles: %d" % (len(_angles), ))
# total number of zero angles
print("Total number of zero angles: %d" % (len(_angles) - np.count_nonzero(_angles)), )

# Build up our Keras model using an Adam optimizer
model = get_model(img_shape)
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

# start our traininng
history = model.fit_generator(
        training_generator(),
        samples_per_epoch=2048,
        nb_epoch=10,
        validation_data=validation_generator(),
        nb_val_samples=256,
)

# save our model
now = datetime.now().isoformat()
model.save('model-%s.h5' % (now, ))
