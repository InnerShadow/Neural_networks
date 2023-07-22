
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten

WHITE_MAX = 255

def show(x_train, N):
	plt.figure(figsize = (10, 5))
	for i in range(N):
		plt.subplot(5, 5, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(x_train[i], cmap = plt.cm.binary)

	plt.show()


def __main__():

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalized
	x_train = x_train / WHITE_MAX
	y_train = y_train / WHITE_MAX

	#show(x_train, 25)

	# Make vectors instad of numbers 
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	model = keras.Sequential([Flatten(input_shape = (28, 28, 1)), 
		Dense(128, activation = 'relu'), Dense(10, activation = 'softmax')])

	print(model.summary())

	

if __name__ == '__main__':
	__main__()
