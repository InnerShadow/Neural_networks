
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten

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

	x_train = x_train / 255
	y_train = y_train / 255

	show(x_train, 25)


if __name__ == '__main__':
	__main__()
