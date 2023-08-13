import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import re

from tensorflow import keras
from keras.layers import Dense, Input, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

maxWordsCount = 10000
max_texts_len = 10

def sequence_to_texts(list_of_indexes, reverce_word_map):
	words = [reverce_word_map.get(letter) for letter in list_of_indexes]
	return (words)


def __main__():

	global maxWordsCount
	global max_texts_len

	characters = np.array(['Reistlin', 'Krisania', 'King-prist', 'Karamon', 'Takhizis'])
	lengts = np.zeros(len(characters), dtype = int)
	texts = []

	for i in range(len(characters)):
		with open(characters[i], 'r', encoding = 'utf-8') as f:
			text = f.readlines()
			text[0] = text[0].replace('\ufeff', '')
			texts += text
			lengts[i] = len(text)

	tokenizer = Tokenizer(num_words = maxWordsCount, filters = '!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»…'
		       , lower = True, split = ' ', char_level = False)
	tokenizer.fit_on_texts(texts)

	try:
		model = load_model('model.h5')
		model.summary()

	except Exception:

		data = tokenizer.texts_to_sequences(texts)
		data_pad = pad_sequences(data, maxlen = max_texts_len)

		X = data_pad
		Y = np.empty((0, len(characters)), dtype = int)
		for i in range(len(characters)):
			arr = np.zeros(len(characters))
			arr[i] = 1
			Y_part = np.array([arr] * lengts[i])
			Y = np.vstack((Y, Y_part))

		indeces = np.random.choice(X.shape[0], size = X.shape[0], replace = False)
		X = X[indeces]
		Y = Y[indeces]

		model = Sequential()
		model.add(Embedding(maxWordsCount, 512, input_length = max_texts_len))
		model.add(LSTM(256, return_sequences = True))
		model.add(LSTM(128))
		model.add(Dense(len(characters), activation = 'softmax'))
		model.summary()

		model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = Adam(0.0001))

		history = model.fit(X, Y, batch_size = 32, epochs = 50)

		model.save('model.h5')

	reverce_word_map = dict(map(reversed, tokenizer.word_index.items()))

	inp = input("Кто сказал: ")
	t = inp.lower()
	data = tokenizer.texts_to_sequences([t])
	data_pad = pad_sequences(data, maxlen = max_texts_len)
	print(sequence_to_texts(data[0], reverce_word_map))

	res = model.predict(data_pad)
	print(res, characters[np.argmax(res)], sep = '\n')


if __name__ == '__main__':
	__main__()

