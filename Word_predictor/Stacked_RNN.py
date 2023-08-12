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
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

inp_words = 3
maxWordsCount = 10000

def buildPhrase(texts, model, tokenizer, str_len = 20):

	global inp_words
	global maxWordsCount

	if len(texts) > inp_words:
		texts = texts[:inp_words]

	res = texts
	data = tokenizer.texts_to_sequences([texts])[0]

	#print(data)

	for i in range(str_len):
		x = data[i : i + inp_words]
		inp = np.expand_dims(x, axis = 0)

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
	res = np.array(data[0])
	print(res.shape)

	n = res.shape[0] - inp_words

	X = np.array([res[i:i + inp_words] for i in range(n)])
	Y = to_categorical(res[inp_words:], num_classes = maxWordsCount)

	input_str = raw_input("Введите начало предоожения: ")

	try :
		model = load_model('Stacked_RNN.h5')
		model.summary()
	except Exception:
		model = Sequential()
		model.add(Embedding(maxWordsCount, 512, input_length = inp_words))
		model.add(SimpleRNN(256, activation = 'tanh', return_sequences = True))
		model.add(SimpleRNN(128, activation = 'tanh'))
		model.add(Dense(maxWordsCount, activation = 'softmax'))

		model.summary()

		model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

		history = model.fit(X, Y, batch_size = 256, epochs = 100)

		model.save('Stacked_RNN.h5')

	res = buildPhrase(input_str, model, tokenizer, 50)
	print(res)


if __name__ == '__main__':
	__main__()

