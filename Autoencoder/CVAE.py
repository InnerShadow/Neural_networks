import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

from PIL import Image
from tensorflow import keras
from keras.layers import Dense, Input, Lambda, Flatten, Reshape, BatchNormalization, Dropout, concatenate
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import mnist

WHITE_MAX = 255
OLD = False # Switch image by image show (True) and all images at screen (False) 

hidden_dim = 2
num_classes = 10
batch_size = 100

z_mean = None
z_log_var = None

def dropout_and_batchnormlizarion(x):
    return Dropout(0.3)(BatchNormalization()(x))


#Random varible generator
def noiser(args):
    global z_mean, z_log_var
    z_mean, z_log_var = args
    N = K.random_normal(shape = (batch_size, hidden_dim), mean = 0., stddev = 1.0)
    return K.exp(z_log_var / 2) * N + z_mean


# x - input data, y - output data
def vae_loss(x, y):
    global batch_size
    x = K.reshape(x, shape = (batch_size, 28 * 28))
    y = K.reshape(y, shape = (batch_size, 28 * 28))
    loss = K.sum(K.square(x - y), axis = -1)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1) # Kullback-Leibler divergence
    return (loss + kl_loss) / 2 / 28 / 28


def showDecoderWork(decoder, number):
    n = 4
    total = 2 * n + 1
    input_lbl = np.zeros((1, num_classes))
    input_lbl[0, number] = 1

    plt.figure(figsize = (total, total))

    h = np.zeros((1, hidden_dim))
    num = 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            ax = plt.subplot(total, total, num)
            num += 1
            h[0, :] = [1 * i / n, 1 * j / n]
            img = decoder.predict([h, input_lbl])
            plt.imshow(img.squeeze(), cmap = 'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


#Show generated images ("num" images per screen)
def plot_digits(*images):
    images = [x.squeeze() for x in images]
    n = min([x.shape[0] for x in images])
    
    plt.figure(figsize = (n, len(images)))
    for j in range(n):
        for i in range(len(images)):
            ax = plt.subplot(len(images), n, i * n + j + 1)
            plt.imshow(images[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


#Show generic image & images of the same hidden layer dimension spot 
def plot_all_images(images):
    num_images = len(images)
    n = len(images[0])

    fig, axis = plt.subplots(num_images, n, figsize = (n, num_images))
    for i, image_list in enumerate(images):
        for j in range(n):
            ax = axis[i, j]
            ax.imshow(image_list[j], cmap = 'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def __mian__():
    global z_log_var
    global hidden_dim, batch_size, num_classes

    #Get train data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / WHITE_MAX
    x_test = x_test / WHITE_MAX

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    #Try to load model
    try:
        cvae = load_model('CVAE.h5', custom_objects = {"vae_loss": vae_loss})
        encoder = load_model('CVAE_encoder.h5')
        decoder = load_model('CVAE_decoder.h5')
        tr_style = load_model('CVAE_tr_style.h5')

        cvae.summary()

    except Exception:

        #Make up codder
        input_img = Input(shape = (28, 28, 1))
        fl = Flatten()(input_img) # Get array from image
        lb = Input(shape = (num_classes,)) # Class mark 
        x = concatenate([fl, lb])
        x = Dense(256, activation = 'relu')(x)
        x = dropout_and_batchnormlizarion(x)
        x = Dense(128, activation = 'relu')(x)
        x = dropout_and_batchnormlizarion(x)

        z_mean2 = Dense(hidden_dim)(x)
        z_log_var = Dense(hidden_dim)(x)

        h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean2, z_log_var])

        #Make up decoder
        input_dec = Input(shape = (hidden_dim, ))
        lb_dec = Input(shape = (num_classes, ))
        d = concatenate([input_dec, lb_dec])
        d = Dense(128, activation = 'relu')(d)
        d = dropout_and_batchnormlizarion(d)
        d = Dense(256, activation = 'relu')(d)
        d = dropout_and_batchnormlizarion(d)
        d = Dense(28 * 28, activation = 'sigmoid')(d)
        decoded = Reshape((28, 28, 1))(d)

        #Consturate model
        encoder = keras.Model([input_img, lb], h, name = 'encoder')
        decoder = keras.Model([input_dec, lb_dec], decoded, name = 'decoder')
        cvae = keras.Model([input_img, lb, lb_dec], decoder([encoder([input_img, lb]), lb_dec]), name = "CVAE")

        #Transfer writing style from one to another number 
        #(From the same spot of hidden dimension spot, but wuth other class mark)
        #(We do not need variance coder output, just mean one).
        z_meaner = keras.Model([input_img, lb], z_mean2)
        tr_style = keras.Model([input_img, lb, lb_dec], decoder([z_meaner([input_img, lb]), lb_dec]), name = 'tr_style')

        cvae.compile(optimizer = 'adam', loss = vae_loss)

        cvae.summary()

        # Train model
        cvae.fit([x_train, y_train_cat, y_train_cat], x_train, epochs = 5, batch_size = batch_size, shuffle = True)

        cvae.save('CVAE.h5')
        encoder.save('CVAE_encoder.h5')
        decoder.save('CVAE_decoder.h5')
        tr_style.save('CVAE_tr_style.h5')

    #Show set of hidden layer spots
    lb = lb_dec = y_test_cat
    h = encoder.predict([x_test, lb], batch_size = batch_size)
    plt.scatter(h[:, 0], h[:, 1])
    plt.show()

    #Test hidden layer dimension
    showDecoderWork(decoder, 2)

    #Test style transfer using z_meaner & tr_style
    dig1 = 5

    num = 10
    X = x_train[y_train == dig1][:num]

    lb_1 = np.zeros((num, num_classes))
    lb_1[:, dig1] = 1

    if(OLD):
        #Image by image show images base on generic image style
        #(Do not representative)
        plot_digits(X)

        for i in range(num_classes):
            lb_2 = np.zeros((num, num_classes))
            lb_2[:, i] = 1

            Y = tr_style.predict([X, lb_1, lb_2], batch_size=num)
            plot_digits(Y)

    else:
        #Show all images on one screen
        image_list = [X]

        for i in range(num_classes):
            lb_2 = np.zeros((num, num_classes))
            lb_2[:, i] = 1

            Y = tr_style.predict([X, lb_1, lb_2], batch_size = num)
            image_list.append(Y)

        plot_all_images(image_list)
    

if __name__ == '__main__':
    __mian__()

