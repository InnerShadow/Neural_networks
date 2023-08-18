
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

WHITE_MAX = 255

def show(x_train, N):
	plt.figure(figsize = (10, 5))
	for i in range(N):
		plt.subplot(5, 5, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(x_train[i], cmap = plt.cm.binary)

	plt.show()


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

	#show(x_train, 25)

	# Make vectors instad of numbers 
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	#make cearten validation set
	size_val = 10000
	x_val_split = x_train[:size_val]
	y_val_split = y_train_cat[:size_val]

	x_train_split = x_train[size_val:]
	y_train_split = y_train_cat[size_val:]

	#For graphs:

	limit = 5000
	x_train_data = x_train[:limit]
	y_train_data = y_train_cat[:limit]

	x_valid = x_train[limit:limit * 2]
	y_valid = y_train_cat[limit:limit * 2]

	#use sklearn
	x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size = 0.2)

	model = keras.Sequential([Flatten(input_shape = (28, 28, 1)), 
		Dense(int(128), activation = 'relu'), Dropout(0.1), Dense(10, activation = 'softmax')])

	print(model.summary())

	myAdam = keras.optimizers.Adam(learning_rate = 0.01)
	myOPT = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0, nesterov = True)

	model.compile(optimizer = myOPT, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	model.fit(x_train, y_train_cat, batch_size = 32, epochs = 5, validation_split = 0.2)

	#model.fit(x_train_split, y_train_split, batch_size = 32, epochs = 5, validation_data = (x_val_split, y_val_split))

	model.evaluate(x_test, y_test_cat)

	index = random.randint(0, 5000)
	x = np.expand_dims(x_test[index], axis = 0)
	res = model.predict(x)

	print("Full vector: ", res)
	print("Guess it is: ", np.argmax(res))

	plt.imshow(x_test[index], cmap = plt.cm.binary)
	plt.show()

	print(model.summary())

	# Predict my own number

	pixel_array_expand, pixel_array = Get_my_immage()

	res = model.predict(pixel_array_expand)
	print("My full vector: ", res)
	print("My gauess it is: ", np.argmax(res))

	plt.imshow(pixel_array, cmap = plt.cm.binary)
	plt.show()

	#Show graphs

	his = model.fit(x_train_data, y_train_data, epochs = 50, batch_size = 32, validation_data = (x_valid, y_valid))

	plt.plot(his.history['loss'])
	plt.plot(his.history['val_loss'])
	plt.show()

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

	for i in range(5):
		print("NN gases: ", p_false[i])
		plt.imshow(x_false[i], cmap = plt.cm.binary)
		plt.show()


if __name__ == '__main__':
	__main__()

