import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Input, Embedding, LSTM, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

N = 10000
M = 200

def __main__():

	global N, M

	data = np.array([np.sin(x / 20) for x in range(N)] + 0.1 * np.random.randn(N))
	plt.plot(data[:100])
	plt.show()

	off = 3
	length = off * 2 + 1
	X = np.array([np.diag(np.hstack((data[i : i + off], data[i + off + 1 : i + length]))) for i in range(N - length)])
	Y = data[off: N - off - 1]
	print(X.shape, Y.shape, sep = '\n')

	model = Sequential()
	model.add(Input((length - 1, length - 1)))
	model.add(Bidirectional(GRU(2)))
	model.add(Dense(1, activation = 'linear'))
	model.summary()

	model.compile(loss = 'mean_squared_error', optimizer = Adam(0.01))

	history = model.fit(X, Y, batch_size = 32, epochs = 10)

	XX = np.zeros(M)
	XX[:off] = data[:off]

	for i in range(M - off - 1):
		x = np.diag(np.hstack((XX[i: i + off], data[i + off + 1: i + length])))
		x = np.expand_dims(x, axis = 0)
		y = model.predict(x)
		XX[i+ off + 1] = y

	plt.plot(data[:M], color = 'b')
	plt.plot(XX[:M], color = 'r')
	plt.show()

if __name__ == '__main__':
	__main__()

