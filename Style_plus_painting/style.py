import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

from PIL import Image
from random import randint
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

B = 103.939
G = 116.779
R = 123.680

def ResizeImg(impath, height, width):
	img = Image.open(impath)
	img = img.resize((height, width))
	img.save(impath)


def processed_image(img, h, w):
	image = img.resize( (h, w), Image.BILINEAR)
	image = np.array(image, dtype = float)
	size = image.shape
	lab = rgb2lab(1.0 / 255 * image)
	X, Y = lab[:, :, 0], lab[:, :, 1:]

	Y /= 128
	X = X.reshape(1, size[0], size[1], 1)
	Y = Y.reshape(1, size[0], size[1], 2)
	return X, Y, size


def load_images(directory, h, w):
    image_paths = glob.glob(os.path.join(directory, "*.jpg"))
    images = []

    for img_path in image_paths:
        img = Image.open(img_path)
        img.resize((h, w))
        X, Y, size = processed_image(img, h, w)	
        images.append((X, Y, size))

    return images


def colarization(input_path, output_path):

	datagen = ImageDataGenerator(
		rotation_range = 5,
		width_shift_range = 0.1,
		height_shift_range = 0.1,
		shear_range = 0.1,
		zoom_range = 0.1,
		fill_mode = 'nearest'
	)

	images = load_images("Train", 680, 680)

	X_data = []
	Y_data = []

	for X, Y, size in images:
		X_data.append(X)
		Y_data.append(Y)

	X_data = np.concatenate(X_data)
	Y_data = np.concatenate(Y_data)
	
	str_img = input_path
	ResizeImg(str_img, 680, 680)
	img = Image.open(str_img)

	X, Y, size = processed_image(img, 680, 680)

	model = Sequential()
	model.add(InputLayer(input_shape = (None, None, 1)))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same', strides = 2))
	model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
	model.add(Conv2D(2, (3, 3), activation = 'tanh', padding = 'same'))
	model.add(UpSampling2D((2, 2)))

	print(model.summary())

	model.compile(optimizer = 'adam', loss = 'mse')

	batch_size = 32
	epochs = 250

	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}/{epochs}")
		for batch in datagen.flow(X_data, Y_data, batch_size = batch_size):
			X_batch, Y_batch = batch
			model.fit(x = X_batch, y = Y_batch, batch_size = batch_size)
			break

	img = Image.open(str_img)
	X, Y, size = processed_image(img, 680, 680)

	output = model.predict(X)

	output *= 128
	min_vals, max_vals = -128, 127
	ab = np.clip(output[0], min_vals, max_vals)

	cur = np.zeros((size[0], size[1], 3))
	cur[:,:,0] = np.clip(X[0][:, :, 0], 0, 100)
	cur[:,:,1:] = ab
	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.subplot(1, 2, 2)
	plt.imshow(lab2rgb(cur))
	plt.show()

	res = lab2rgb(cur)
	res = (res * 255).astype(np.uint8)
	imsave(output_path, res)


def ResizeImg(path, height, width):
	img = Image.open(path)
	img = img.resize((height, width))
	img.save(path)


def deprecess_image(precessed_image):
	x = precessed_image.copy()
	if len(x.shape) == 4:
		x = np.squeeze(x, 0)
	assert len(x.shape) == 3, ('Wrong image dims')

	if len(x.shape) != 3:
		raise ValueError("Invalid input")

	x[:, :, 0] += B
	x[:, :, 1] += G
	x[:, :, 2] += R

	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')
	return x


def get_feature_representation(model, x_img, x_style, num_style_layers, num_content_layers):
	style_outputs = model(x_style)
	content_outputs = model(x_img)

	style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
	content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
	return style_features, content_features


def get_content_loss(base_content, target):
	return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
	channels = int(input_tensor.shape[-1])
	a = tf.reshape(input_tensor, [-1, channels])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a = True)
	return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
	gram_style = gram_matrix(base_style)

	return tf.reduce_mean(tf.square(gram_style - gram_target))


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, 
	num_style_layers, num_content_layers):
	style_weights, content_weights = loss_weights
	model_outputs = model(init_image)

	style_output_features = model_outputs[:num_style_layers]
	content_outputs_features = model_outputs[num_style_layers:]

	style_score = 0
	content_score = 0 

	weight_per_style_layer = 1.0 / float(num_style_layers)
	for target_style, comb_style in zip(gram_style_features, style_output_features):
		style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

	weight_per_content_layer = 1.0 / float(num_content_layers)
	for target_content, comb_content in zip(content_features, content_outputs_features):
		content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

	style_score *= style_weights
	content_score *= content_weights

	loss = style_score + content_score

	return loss, style_score, content_score


def get_style(input_path_img, imput_path_style, output_path):
	str_img = input_path_img
	str_img_style = imput_path_style

	height, width = (680, 680)

	ResizeImg(str_img, height, width)
	ResizeImg(str_img_style, height, width)

	img = Image.open(str_img)
	img_style = Image.open(str_img_style)

	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.subplot(1, 2, 2)
	plt.imshow(img_style)
	plt.show()

	x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis = 0))
	x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis = 0))

	content_layers = ['block5_conv2']

	style_layers = ['block1_conv1', 
	'block2_conv1', 
	'block3_conv1',
	'block4_conv1',
	'block5_conv1']

	num_content_layers = len(content_layers)
	num_style_layers = len(style_layers)

	vgg = keras.applications.vgg19.VGG19(include_top = False, weights = 'imagenet')
	vgg.trainable = False

	style_output = [vgg.get_layer(name).output for name in style_layers]
	content_outputs = [vgg.get_layer(name).output for name in content_layers]
	model_outputs = style_output + content_outputs

	print(vgg.input)
	for m in model_outputs:
		print(m)

	model = keras.models.Model(vgg.input, model_outputs)
	print(model.summary())

	num_iterators = 25
	content_weights = 1e3
	style_weight = 1e-2

	style_features, content_features = get_feature_representation(model, x_img, x_style, num_style_layers, num_content_layers)
	gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

	init_image = np.copy(x_img)
	init_image = tf.Variable(init_image, dtype = tf.float32)

	opt = tf.compat.v1.train.AdamOptimizer(learning_rate = 2, beta1 = 0.99, epsilon = 1e-1)
	irer_count = 1
	best_loss, best_img = float('inf'), None
	loss_weights = (style_weight, content_weights)

	cfg = {
		'model': model,
		'loss_weights': loss_weights,
		'init_image': init_image,
		'gram_style_features' : gram_style_features,
		'content_features': content_features,
		'num_style_layers': num_style_layers,
		'num_content_layers': num_content_layers
	}

	norm_mean = np.array([B, G, R])
	min_vals = -norm_mean
	max_vals = 255 - norm_mean
	imgs = []

	for i in range(num_iterators):
		with tf.GradientTape() as tape:
			all_loss = compute_loss(**cfg)

		total_loss = all_loss[0]
		grads = tape.gradient(total_loss, init_image)

		loss, style_score, content_score = all_loss
		opt.apply_gradients([(grads, init_image)])
		clipped = tf.clip_by_value(init_image, min_vals, max_vals)
		init_image.assign(clipped)

		if loss < best_loss:
			best_loss = loss
			best_img = deprecess_image(init_image.numpy())

			plot_image = deprecess_image(init_image.numpy())
			imgs.append(plot_image)
			print('Iteration: {}'.format(i))


	plt.imshow(best_img)
	plt.show()
	print(best_loss)

	for it in range(len(imgs)):
		image = Image.fromarray(imgs[it].astype('uint8'), 'RGB')
		image.save(str("step_by_step/") + str(it) + str("res.jpg"))

	image = Image.fromarray(best_img.astype('uint8'), 'RGB')
	image.save(output_path)


def __main__():
	get_style("img.jpg", "img_style.jpg", "res0.jpg")
	get_style("res0.jpg", "img_style.jpg", "res1.jpg")
	get_style("res1.jpg", "img_style.jpg", "res2.jpg")
	get_style("res2.jpg", "img_style.jpg", "res3.jpg")
	get_style("res3.jpg", "img_style.jpg", "res4.jpg")
	colarization("res4.jpg", "res_final.jpg")


if __name__ == '__main__':
	__main__()

