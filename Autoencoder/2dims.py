import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from keras.layers import Dense, Input, Embedding, Flatten, Reshape
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import mnist

WHITE_MAX = 255

def __mian__():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / WHITE_MAX
    x_test = x_test / WHITE_MAX

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

    img_input = Input((28, 28, 1))
    x = Flatten()(img_input)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    encoded = Dense(2, activation = 'linear')(x) # hidden

    input_enc = Input(shape = (2, ))
    d = Dense(64, activation = 'relu')(input_enc)
    d = Dense(28 * 28, activation = 'sigmoid')(d)
    decoded = Reshape((28, 28, 1))(d)

    encoder = keras.Model(img_input, encoded, name = 'encoder')
    decoder = keras.Model(input_enc, decoded, name = 'decoder')
    autoencoder = keras.Model(img_input, decoder(encoder(img_input)), name = 'autoencoder')
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

    try:
        autoencoder = load_model('2dims.h5')
        autoencoder.summary()
        
    except Exception:

        autoencoder.summary()

        batch_size = 100    

        autoencoder.fit(x_train, x_train, epochs = 20, batch_size = batch_size, shuffle = True)

        autoencoder.save('2dims.h5')

    h = encoder.predict(x_test)

    #Show set of hidden layer spots

    a = plt.scatter(h[:, 0], h[:, 1])
    plt.show()

    #Show [0, 0] vector & [10, 10] vector

    img = decoder.predict(np.expand_dims([0, 0], axis = 0))
    plt.imshow(img.squeeze(), cmap = 'gray')
    plt.show()

    img = decoder.predict(np.expand_dims([10, 10], axis = 0))
    plt.imshow(img.squeeze(), cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    __mian__()

