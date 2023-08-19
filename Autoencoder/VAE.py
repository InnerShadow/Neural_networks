import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

from PIL import Image
from tensorflow import keras
from keras.layers import Dense, Input, Lambda, Flatten, Reshape, BatchNormalization, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import mnist

WHITE_MAX = 255

hidden_dims = 2
batch_size = 60

def dropout_and_batchnormlizarion(x):
    return Dropout(0.3)(BatchNormalization()(x))


def noiser(args):
    global hidden_dims, batch_size
    z_mean, z_log_var = args
    N = K.random_normal(shape = (batch_size, hidden_dims), mean = 0., stddev = 1.0)
    return K.exp(z_log_var / 2) * N + z_mean


def __mian__():
    global hidden_dims, batch_size

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / WHITE_MAX
    x_test = x_test / WHITE_MAX

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    #Make up codder
    img_input = Input(batch_size = (batch_size, 28, 28, 1))
    x = Flatten()(img_input)
    x = Dense(256, activation = 'relu')(x)
    x = dropout_and_batchnormlizarion(x)
    x = Dense(128, activation = 'relu')(x)
    x = dropout_and_batchnormlizarion(x)

    z_mean = Dense(hidden_dims)(x)
    z_log_var = Dense(hidden_dims)(x)

    h = Lambda(noiser, output_shape = (hidden_dims,))([z_mean, z_log_var])

    #Make up decoder
    input_dec = Input(shape = (hidden_dims,))
    d = Dense(128, activation = 'relu')(input_dec)
    d = dropout_and_batchnormlizarion(d)
    d = Dense(256, activation = 'relu')(input_dec)
    d = dropout_and_batchnormlizarion(d)
    d = Dense(28 * 28, activation = 'sigmoid')(d)
    decoded = Reshape((28, 28, 1))(d)

    encoder = keras.Model(img_input, h, name = 'encoder')
    decoder = keras.Model(input_dec, decoded, name = 'decoder')
    vae = keras.Model(img_input, decoder(encoder(img_input)), name = 'VAE')


if __name__ == '__main__':
    __mian__()
