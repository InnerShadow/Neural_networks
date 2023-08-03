import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

WHITE_MAX = 255


def Get_my_immage(image_path):
	image = Image.open(image_path)

	gary_image = image.convert("L")

	pixel_array = np.array(gary_image)

	pixel_array = WHITE_MAX - pixel_array
	pixel_array = pixel_array / WHITE_MAX

	pixel_array_expand = np.expand_dims(pixel_array, axis = 0)

	return pixel_array_expand, pixel_array


def Predict_number_by_path(model, image_path):

	pixel_array_expand, pixel_array = Get_my_immage(image_path)

	res = model.predict(pixel_array_expand)
	print("My full vector: ", res)
	print("My gauess it is: ", np.argmax(res))

	plt.imshow(pixel_array, cmap = plt.cm.binary)
	plt.show()

def __main__():

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalized
	x_train = x_train / WHITE_MAX
	x_test = x_test / WHITE_MAX

	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	x_train = np.expand_dims(x_train, axis = 3)
	x_test = np.expand_dims(x_test, axis = 3)

	model = keras.Sequential([
		Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)),
		MaxPooling2D((2, 2), strides = 2),
		Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
		MaxPooling2D((2, 2), strides = 2),
		Flatten(),
		Dense(128, activation = 'relu'),
		Dense(10, activation = 'softmax')])

	print(model.summary())

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	his = model.fit(x_train, y_train_cat, batch_size = 32 ,epochs = 5, validation_split = 0.2)

	model.evaluate(x_test, y_test_cat)

	Predict_number_by_path(model, "test.png")
	Predict_number_by_path(model, "Nazar_test_2.png")

if __name__ == '__main__':
	__main__()
