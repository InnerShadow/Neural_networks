import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import re

from PIL import Image
from random import randint
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_characters = 40

def buildPhrase(inp_str, inp_chars, model, tokenizer, str_len = 200):

	global num_characters

	for i in range(str_len):
		x = []
		for j in range(i, i + inp_chars):
			x.append(tokenizer.texts_to_matrix([inp_str[j]]))

		x = np.array(x)
		inp = x.reshape(1, inp_chars, num_characters)

		pred = model.predict(inp)
		d = tokenizer.index_word[pred.argmax(axis = 1)[0]]

		inp_str += d

	return inp_str


def __main__():

	global num_characters

	with open('Last_Challenge', 'r', encoding = 'utf-8') as f:
		text = f.read()
		test = text.replace('\ufeff', ' ')
		text = re.sub(r'[^А-я –.,?!:]', '', text)

	tokenizer = Tokenizer(num_words = num_characters, char_level = True)
	tokenizer.fit_on_texts([text])
	print(tokenizer.word_index)

	input_str = 'власть н'
	inp_chars = len(input_str)
	data = tokenizer.texts_to_matrix(text)
	n = data.shape[0] - inp_chars

	X = np.array([data[i:i + inp_chars, :] for i in range(n)])
	Y = data[inp_chars:]

	try:
		model = load_model('model.h5')
		model.summary()
	except Exception:
		model = Sequential()
		model.add(Input((inp_chars, num_characters)))
		model.add(SimpleRNN(512, activation = 'tanh'))
		model.add(Dense(256, activation = 'relu'))
		model.add(Dense(128, activation = 'relu'))
		model.add(Dense(num_characters, activation = 'softmax'))

		model.summary()

		model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

		history = model.fit(X, Y, batch_size = 350, epochs = 100)

		model.save('model.h5')

	res = buildPhrase(input_str, inp_chars, model, tokenizer, str_len = 200)
	print(res)


if __name__ == '__main__':
	__main__()

