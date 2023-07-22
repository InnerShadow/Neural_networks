
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
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
	x_test = x_test / WHITE_MAX

	show(x_train, 25)

	# Make vectors instad of numbers 
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	model = keras.Sequential([Flatten(input_shape = (28, 28, 1)), 
		Dense(128 * 2, activation = 'relu'), Dense(10, activation = 'softmax')])

	print(model.summary())

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	model.fit(x_train, y_train_cat, batch_size = 32, epochs = 5, validation_split = 0.2)

	model.evaluate(x_test, y_test_cat)

	index = random.randint(0, 5000)
	x = np.expand_dims(x_test[index], axis = 0)
	res = model.predict(x)
	print("Full vector: ", res)
	print("Gases it is: ", np.argmax(res))

	plt.imshow(x_test[index], cmap = plt.cm.binary)
	plt.show()

	print(model.summary())

	# Show how it works:

	pred = model.predict(x_test)
	pred = np.argmax(pred, axis = 1)

	print(pred.shape)

	print(pred[:20])
	print(y_test[:20])

	# Show errors:

	mask = pred == y_test
	print(mask[:10])

	x_false = x_test[~mask]
	p_false = pred[~mask]

	print(x_false.shape)

	for i in range(5):
		print("NN gases: ", p_false[i])
		plt.imshow(x_false[i], cmap = plt.cm.binary)
		plt.show()


if __name__ == '__main__':
	__main__()
