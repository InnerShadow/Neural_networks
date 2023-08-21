import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import time

from PIL import Image
from tensorflow import keras
from keras.layers import Dense, Input, Lambda, Flatten, Reshape, BatchNormalization, Dropout, concatenate
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import mnist

WHITE_MAX = 255
BUFFER_SIZE = 0
BATCH_SIZE = 100
EPOCHS = 20 

hidden_dim = 2

cross_entropy = keras.losses.BinaryCrossentropy(from_logits = True)

#Count loss
def generator_loss(fake_out):
    loss = cross_entropy(tf.ones_like(fake_out), fake_out)
    return loss


def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    total_loss = real_loss + fake_loss
    return total_loss


#Train tf 2.0 & applay gradients
@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizzer):
    noise = tf.random.normal([BATCH_SIZE, hidden_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)

        real_ouput = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_ouput, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizzer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


#Start train function
def start_train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizzer):
    history = []
    MAX_PRINT_LABEL = 10
    th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

    for epoch in range(1, epochs + 1):
        print(f'{epoch}/{epochs}:', end = '')

        start = time.time()
        n = 0

        gen_loss_epoch = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizzer)
            gen_loss_epoch += K.mean(gen_loss)
            if (n % th == 0) : print ('=', end = '')
            n += 1

        history += [gen_loss_epoch / n]
        print(': ' + str(history[-1]))
        print('{} epoch time is {} seconds'.format(epoch, time.time() - start))

    return history


def __mian__():
    global BATCH_SIZE, BUFFER_SIZE

    number = 7

    #Get train data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[y_train == number]
    y_train = y_train[y_train == number]

    BUFFER_SIZE = x_train.shape[0]

    BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
    x_train = x_train[:BUFFER_SIZE]
    y_train = y_train[:BUFFER_SIZE]
    #print(x_train.shape(), y_train.shape)

    #Standarsize data
    x_train = x_train / WHITE_MAX
    x_test = x_test / WHITE_MAX

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    #Get shuffeld "BATCH_SIZE" batcher with "number"
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    try:
        generator = load_model('GAN_generator.h5')
        discriminator = load_model('GAN_discriminator.h5')

        generator.summary()
        discriminator.summary()

    except:

        #Make up generator
        generator = Sequential([
            Dense(7 * 7 * 256, activation = 'relu', input_shape = (hidden_dim, )),
            BatchNormalization(),
            Reshape((7, 7, 256)),
            Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'),
            BatchNormalization(),
            Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same', activation = 'relu'),
            BatchNormalization(),
            Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', activation = 'sigmoid'),
        ])

        #Make up Discriminator
        discriminator = Sequential()
        discriminator.add(Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [28, 28, 1]))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.3))
        discriminator.add(Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        discriminator.add(LeakyReLU())
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        
        generator_optimizer = Adam(1e-4)
        discriminator_optimizzer = Adam(1e-4)

        generator.summary()
        discriminator.summary()

        #Do train
        history = start_train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizzer)

        generator.save('GAN_generator.h5')
        discriminator.save('GAN_discriminator.h5')

        #Show loss history
        plt.plot(history)
        plt.grid(True)
        plt.show()

    #Show generation resualts
    n = 2
    total = 2 * n + 1

    plt.figure(figsize = (total, total))

    num = 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            ax = plt.subplot(total, total, num)
            num += 1
            img = generator.predict(np.expand_dims([0.5 * i / n, 0.5 * j / n], axis = 0))
            plt.imshow(img[0, :, :, 0], cmap = 'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()
    

if __name__ == '__main__':
    __mian__()

