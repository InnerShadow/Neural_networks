
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense

celsia = np.array([-40, -10, 0, 8, 15, 22, 38])
faringats = np.array([-40, 14, 32, 46, 59, 72, 100])


def __main__():
	model = keras.Sequential()
	model.add(Dense(units = 1, input_shape = (1, ), activation = 'linear'))
	model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.1))

	log = model.fit(celsia, faringats, epochs = 500, verbose = False)
	print("Ends training")

	print(model.predict([100]))
	print(model.get_weights())

	plt.plot(log.history['loss'])
	plt.grid(True)
	plt.show()


if __name__ == '__main__':
	__main__()

