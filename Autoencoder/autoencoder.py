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

def Get_my_immage(image_path):
	image = Image.open(image_path)

	gary_image = image.convert("L")

	pixel_array = np.array(gary_image)

	pixel_array = WHITE_MAX - pixel_array
	pixel_array = pixel_array / WHITE_MAX

	pixel_array_expand = np.expand_dims(pixel_array, axis = 0)

	return pixel_array_expand, pixel_array


def plot_digits(*images):
    images = [x.squeeze() for x in images]
    n = images[0].shape[0] 

    plt.figure(figsize = (n, 1))
    for j in range(n):
        ax = plt.subplot(1, n, j + 1)
        plt.imshow(images[0][j])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_homotopy(frm, to, n = 10, autoencoder = None):
    z = np.zeros(([n] + list(frm.shape)))
    for i, t in enumerate(np.linspace(0., 1., n)):
        z[i] = frm * (1 - t) + to * t # direct homotopy
    if autoencoder :
        plot_digits(autoencoder.predict(z, batch_size = n))
    else:
        plot_digits(z)


def __mian__():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / WHITE_MAX
    x_test = x_test / WHITE_MAX

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

    try:
        autoencoder = load_model('autoencoder.h5')
        autoencoder.summary()
        
    except Exception:

        img_input = Input((28, 28, 1))
        x = Flatten()(img_input)
        x = Dense(128, activation = 'relu')(x)
        x = Dense(63, activation = 'relu')(x)
        encoded = Dense(49, activation = 'relu')(x) # hidden

        d = Dense(64, activation = 'relu')(encoded)
        d = Dense(28 * 28, activation = 'sigmoid')(d)
        decoded = Reshape((28, 28, 1))(d)

        autoencoder = keras.Model(img_input, decoded, name = 'autoencoder')
        autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

        autoencoder.summary()

        batch_size = 100

        autoencoder.fit(x_train, x_train, epochs = 20, batch_size = batch_size, shuffle = True)

        autoencoder.save('autoencoder.h5')

    #show my image

    pixel_array_expand, pixel_array = Get_my_immage("Nazar_test_3.png")

    res = autoencoder.predict(pixel_array_expand)

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].imshow(pixel_array, cmap = 'gray')
    axes[0].set_title('Original images')

    axes[1].imshow(res.squeeze(), cmap = 'gray')
    axes[1].set_title('Reconstucted image')

    plt.tight_layout()
    plt.show()

    #show

    n = 10
    imgs = x_test[:n]
    decoded_imgs = autoencoder.predict(x_test[:n], batch_size = n)

    plt.figure(figsize = (n, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap = 'gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax2 = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i].squeeze(), cmap = 'gray')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
    plt.show()

    # show difference between direct himotopy and model that build NN

    frm, to = x_test[y_test == 5][1:3]
    plot_homotopy(frm, to)
    plot_homotopy(frm, to, autoencoder = autoencoder)


if __name__ == '__main__':
    __mian__()

