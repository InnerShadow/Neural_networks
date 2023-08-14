import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import re

from tensorflow import keras
from keras.layers import Dense, Embedding, GRU
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from PIL import Image

from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split

maxWordsCount = 10000
max_texts_len = 10

def sequence_to_texts(list_of_indexes, reverce_word_map):
	words = [reverce_word_map.get(letter) for letter in list_of_indexes]
	return (words)


def __main__():

	global maxWordsCount
	global max_texts_len

	with open('true', 'r', encoding = 'utf-8') as f:
		texts_true = f.readlines()
		texts_true[0] = texts_true[0].replace('\ufeff', '')
		
	with open('false', 'r', encoding = 'utf-8') as f:
		texts_false = f.readlines()
		texts_false[0] = texts_false[0].replace('\ufeff', '')
	
	texts = texts_true + texts_false
	count_true = len(texts_true)
	count_false = len(texts_false)
	total_len = count_true + count_false
	print(count_true, count_false, total_len)

	tokenizer = Tokenizer(num_words = maxWordsCount, filters = '!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»…'
		       , lower = True, split = ' ', char_level = False)
	tokenizer.fit_on_texts(texts)
	
	dlist = list(tokenizer.word_counts.items())
	data = tokenizer.texts_to_sequences(texts)
	data_pad = pad_sequences(data, maxlen = max_texts_len)

	#print(dlist[:10])
	#print(texts[0][:100])
	
	#print(data_pad)
	#print(	list(tokenizer.word_index.items()))

	X = data_pad
	Y = np.array([[1, 0]] * count_true + [[0, 1]] * count_false)
	
	indeces = np.random.choice(X.shape[0], size = X.shape[0], replace = False)
	X = X[indeces]
	Y = Y[indeces]

	try:
		model = load_model('GRU_model.h5')
		model.summary()

	except Exception:

		model = Sequential()
		model.add(Embedding(maxWordsCount, 128, input_length = max_texts_len))
		model.add(GRU(128, return_sequences = True))
		model.add(GRU(64))
		model.add(Dense(2, activation = 'softmax'))
		model.summary()

		model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = Adam(0.0001))

		history = model.fit(X, Y, batch_size = 64, epochs = 50)

		model.save('GRU_model.h5')

	reverce_word_map = dict(map(reversed, tokenizer.word_index.items()))

	inp = input("Введите высказывание: ")
	t = inp.lower()
	data = tokenizer.texts_to_sequences([t])
	data_pad = pad_sequences(data, maxlen = max_texts_len)
	print(sequence_to_texts(data[0], reverce_word_map))

	res = model.predict(data_pad)
	print(res, np.argmax(res), sep = '\n')


if __name__ == '__main__':
	__main__()

