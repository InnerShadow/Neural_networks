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
from tensorflow.keras.utils import to_categorical

inp_words = 3
maxWordsCount = 5000

def buildPhrase(texts, model, tokenizer, str_len = 20):

	global inp_words
	global maxWordsCount

	res = texts
	data = tokenizer.texts_to_sequences([texts])[0]

	for i in range(str_len):
		x = to_categorical(data[i: i + inp_words], num_classes = maxWordsCount)
		inp = x.reshape(1, inp_words, maxWordsCount)

		pred = model.predict(inp)
		indx = pred.argmax(axis = 1)[0]
		data.append(indx)

		res += " " + tokenizer.index_word[indx]
		
	return res


def __main__():

	global inp_words
	global maxWordsCount

	with open('Last_Challenge', 'r', encoding = 'utf-8') as f:
		texts = f.read()
		texts = texts.replace('\ufeff', '')

	tokenizer = Tokenizer(num_words = maxWordsCount, filters = '!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»…',
	lower = True, split = ' ', char_level = False)
	tokenizer.fit_on_texts([texts])

	# dlist = list(tokenizer.word_counts.items())
	# print(dlist[:100])

	data = tokenizer.texts_to_sequences([texts])
	res = to_categorical(data[0], num_classes = maxWordsCount)
	print(res.shape)

	n = res.shape[0] - inp_words

	X = np.array([res[i:i + inp_words, :] for i in range(n)])
	Y = res[inp_words:]

	model = Sequential()
	model.add(Input((inp_words, maxWordsCount)))
	model.add(SimpleRNN(512, activation = 'tanh'))
	model.add(Dense(256, activation = 'relu'))
	model.add(Dense(maxWordsCount, activation = 'softmax'))

	model.summary()

	model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

	history = model.fit(X, Y, batch_size = 256, epochs = 100)

	res = buildPhrase("Маджере что сделать", model, tokenizer, 50)
	print(res)



if __name__ == '__main__':
	__main__()

