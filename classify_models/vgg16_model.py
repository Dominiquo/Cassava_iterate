# Code adapted via Benjamin Akera and Hvass-labs Tutorial
# https://github.com/BenjaminAkera/
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop


def get_transfer_model(num_classes, batch_size=20):
	model = VGG16(include_top=True, weights='imagenet')
	input_shape = model.layers[0].output_shape[1:3]
	transfer_layer = model.get_layer('block5_pool')
	conv_model = Model(inputs=model.input,
	                   outputs=transfer_layer.output)

	# Start a new Keras Sequential model.
	new_model = Sequential()
	# Add the convolutional part of the VGG16 model from above.
	new_model.add(conv_model)
	# Flatten the output of the VGG16 model because it is from a
	# convolutional layer.
	new_model.add(Flatten())
	# Add a dense (aka. fully-connected) layer.
	# This is for combining features that the VGG16 model has
	# recognized in the image.
	new_model.add(Dense(1024, activation='relu'))
	# Add a dropout-layer which may prevent overfitting and
	# improve generalization ability to unseen data e.g. the test-set.
	new_model.add(Dropout(0.5))
	# Add the final layer for the actual classification.
	new_model.add(Dense(num_classes, activation='softmax'))

	optimizer = Adam(lr=1e-5)
	loss = 'categorical_crossentropy'
	metrics = ['categorical_accuracy']

	conv_model.trainable = False
	for layer in conv_model.layers:
	    layer.trainable = False

	new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return new_model



