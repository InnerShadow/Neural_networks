import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from random import randint
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ResizeImg(img, height, width, impath):
	img = img.resize((height, width))
	img.save(impath)


def processed_image(img, h, w):
	image = img.resize( (h, w), Image.BILINEAR)
	image = np.array(image, dtype = float)
	size = image.shape
	lab = rgb2lab(1.0 / 255 * image)
	X, Y = lab[:, :, 0], lab[:, :, 1:]

	Y /= 128
	X = X.reshape(1, size[0], size[1], 1)
	Y = Y.reshape(1, size[0], size[1], 2)
	return X, Y, size


def load_images(directory, h, w):
	image_paths = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
	images = []

	for img_path in image_paths:
		img = Image.open(os.path.join(directory, img_path))
		ResizeImg(img, w, h, os.path.join(directory, img_path))
		X, Y, size = processed_image(img, h, w)
		images.append((X, Y, size))

	return images


def __main__():
	
	directory = 'Train'
	image_data = load_images(directory, 256, 256)

	input_images = []
	output_images = []

	for X, Y, size in image_data:
		input_images.append(X)
		output_images.append(Y)

	X_data = np.concatenate(input_images, axis = 0)
	Y_data = np.concatenate(output_images, axis = 0)

	data_generator = ImageDataGenerator(
	    rotation_range = 5,       
	    width_shift_range = 0.05,   
	    height_shift_range = 0.05, 
	    brightness_range = (0.8, 1.0), 
	    #horizontal_flip = True,    
	    rescale = 1.0 / 255
	)

	augmented_data = data_generator.flow(X_data, Y_data, batch_size = 2000)

	model = Sequential()
	model.add(InputLayer(input_shape = (None, None, 1)))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(2, (3, 3), activation = 'tanh', padding = 'same'))
	model.add(UpSampling2D((2, 2)))

	print(model.summary())

	model.compile(optimizer = 'adam', loss = 'mse')
	model.fit(augmented_data, epochs = 20)

	index = randint(0, len(image_data) - 1)
	x = image_data[index]

	output = model.predict(x[0])

	output *= 128
	min_vals, max_vals = -128, 127
	ab = np.clip(output[0], min_vals, max_vals)

	cur = np.zeros((size[0], size[1], 3))
	cur[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)
	cur[:, :, 1:] = ab
	# plt.subplot(1, 2, 1)
	# plt.imshow(image_data[index])
	# plt.subplot(1, 2, 2)
	plt.imshow(lab2rgb(cur))
	plt.show()


if __name__ == '__main__':
	__main__()

