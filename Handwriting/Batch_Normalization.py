
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

WHITE_MAX = 255


def Get_my_immage():
	image_path = "Nazar_test_3.png"
	image = Image.open(image_path)

	gary_image = image.convert("L")

	pixel_array = np.array(gary_image)

	pixel_array = WHITE_MAX - pixel_array
	pixel_array = pixel_array / WHITE_MAX

	pixel_array_expand = np.expand_dims(pixel_array, axis = 0)

	return pixel_array_expand, pixel_array


def __main__():

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalized
	x_train = x_train / WHITE_MAX
	x_test = x_test / WHITE_MAX

	# Make vectors instad of numbers 
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	limit = 5000
	x_train_data = x_train[:limit]
	y_train_data = y_train_cat[:limit]

	x_valid = x_train[limit:limit * 2]
	y_valid = y_train_cat[limit:limit * 2]

	model = keras.Sequential([Flatten(input_shape = (28, 28, 1)), 
		Dense(int(300), activation = 'relu'), Dense(10, activation = 'softmax')])

	myAdam = keras.optimizers.Adam(learning_rate = 0.01)
	myOPT = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0, nesterov = True)

	model.compile(optimizer = myOPT, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	#Show graphs for 50 epochs

	his = model.fit(x_train_data, y_train_data, epochs = 50, batch_size = 32, validation_data = (x_valid, y_valid))

	plt.plot(his.history['loss'])
	plt.plot(his.history['val_loss'])
	plt.show()

if __name__ == '__main__':
	__main__()

