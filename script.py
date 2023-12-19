#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# This function 'convert_samples' takes a sample from a dataset, which is assumed to have
# a dictionary-like structure with keys 'image' and 'label'. The purpose of this function
# is to preprocess the image and label data before it is used in a machine learning model
def convert_sample(sample):
    # Extract the 'image' and 'label' from the sample
    image, label = sample['image'], sample['label']

    # Convert the image to a floating-point format (tf.float32)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Convert the label to one-hot encoding with 2 classes (assuming binary classification)
    label = tf.one_hot(label, 2, dtype=tf.float32)

    # Return the preprocessed image and label
    return image, label

# Loading the PCam dataset using Tensorflow datasets (tfds)
ds1, ds2, ds3 = tfds.load('patch_camelyon', split=['train[:20%]', 'test[:5%]', 'validation[:5%]'],
                         data_dir='/content/drive/MyDrive',
                         download=True,
                         shuffle_files=True)

# %%
