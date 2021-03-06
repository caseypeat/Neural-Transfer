import numpy as np
import tensorflow as tf
import cv2
import os

from tqdm import tqdm

from tensorflow.train import AdamOptimizer

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model


# Loss functions
def gram_matrix(input_tensor):
	""" Calculates the gram matrix of a tensor """

	vector = tf.reshape(input_tensor, [-1, tf.shape(input_tensor)[3]])

	gram = tf.matmul(vector, vector, transpose_a=True)

	return gram

def calc_style_loss(style, combination):
	""" Calculates the style loss between two output featuremap tensors """

	style_gram = gram_matrix(style)
	combination_gram = gram_matrix(combination)

	mse = tf.reduce_sum(tf.square(style_gram - combination_gram))
	normilize_constant = tf.square(tf.reduce_prod(tf.cast(tf.shape(combination), dtype=tf.float32)))

	loss = tf.divide(mse, normilize_constant)
	loss_reshaped = tf.expand_dims(loss, axis=0)

	return loss_reshaped

def calc_content_loss(content, combination):
	""" Calculates the content loss between two output featuremap tensors """

	mse = tf.reduce_sum(tf.square(content - combination))
	normilize_constant = tf.reduce_prod(tf.cast(tf.shape(combination), dtype=tf.float32))

	loss = tf.divide(mse, normilize_constant)
	loss_reshaped = tf.expand_dims(loss, axis=0)

	return loss_reshaped

def calc_total_loss(content_featuremaps, style_featuremaps, combination_featuremaps, alpha, beta):
	""" Calculates total loss over all style and content output featuremap tensors """

	total_loss = 0

	content_losses_list = []
	style_losses_list = []

	# Iterate over featuremaps for matching content
	for content, combination in zip(content_featuremaps, combination_featuremaps[:len(content_featuremaps)]):

		loss = calc_content_loss(content, combination) * alpha
		total_loss += loss
		content_losses_list.append(loss)	# record individual loss for analysis

	# Iterate over featuremaps for matching style
	for style, combination in zip(style_featuremaps, combination_featuremaps[len(content_featuremaps):]):

		loss = calc_style_loss(style, combination) * beta
		total_loss += loss
		style_losses_list.append(loss)	 # record individual loss for analysis

	content_losses = np.array(content_losses_list)
	style_losses = np.array(style_losses_list)

	return total_loss, content_losses, style_losses



# Initiate models
def init_model(content_layers, style_layers):
	""" Return base model (VGG) with outputs for layers used in loss calculations """

	base_model = VGG19(include_top=False, weights='imagenet')

	outputs = [base_model.get_layer(output_layer).output for output_layer in content_layers+style_layers]

	model = Model(inputs=base_model.inputs, outputs=outputs)

	model.trainable = False

	return model



def neural_transfer(content_image, style_image, output_dirpath, epochs=1000, epoch_length=100, alpha=1, beta=10):
	""" Main function to execute neural transfer algorithm using tensorflow eager execution """

	tf.enable_eager_execution()

	optimizer = AdamOptimizer(learning_rate=0.003)

	# Layers for loss calculations
	content_layers = ['block4_conv2']
	style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

	model = init_model(content_layers, style_layers)

	# Get target featuremaps tensors from the content and style images
	content_featuremaps = model(np.expand_dims(content_image, axis=0))[:len(content_layers)]
	style_featuremaps = model(np.expand_dims(style_image, axis=0))[len(content_layers):]

	# Starting point of combination image
	image_zero = np.expand_dims(np.random.random(np.shape(content_image)), axis=0)
	combined_image_tensor = tf.Variable(image_zero, name='combined_image_tensor', dtype=tf.float32)

	for epoch in range(epochs):

		print('\nEpoch: ', epoch)

		# Convert tensor to array then save image to output directory for viewing
		combined_image = np.squeeze(combined_image_tensor.numpy(), axis=0)
		output_filepath = os.path.join(output_dirpath, 'epoch_{}.png'.format(epoch))
		cv2.imwrite(output_filepath, combined_image * 255)

		content_losses_array_avg = np.zeros(len(content_layers), dtype=np.float32)
		style_losses_array_avg = np.zeros(len(style_layers), dtype=np.float32)

		for _ in tqdm(range(epoch_length)):

			# Operations here are recorded to "GradientTape" for backpropagation
			with tf.GradientTape() as tape:

				combination_featuremaps = model(combined_image_tensor)

				total_loss, content_losses, style_losses = calc_total_loss(content_featuremaps, style_featuremaps, combination_featuremaps, alpha, beta)
				
			gradients = tape.gradient(total_loss, combined_image_tensor)
			optimizer.apply_gradients([[gradients, combined_image_tensor]])

			# Ensure output image/tensor is bounded between 0 and 1
			clipped = tf.clip_by_value(combined_image_tensor, clip_value_min=0, clip_value_max=1)
			combined_image_tensor.assign(clipped)

			# Record the average losses for the epoch
			content_losses_array_avg += content_losses / epoch_length
			style_losses_array_avg += style_losses / epoch_length

		# Display individual losses for analysis
		print('Content loss: ', content_losses_array_avg)
		print('Style loss: ', style_losses_array_avg)
		print('Total loss: ', np.sum(style_losses_array_avg) + np.sum(content_losses_array_avg))



if __name__ == '__main__':

	content_image = cv2.imread('./input_images/european_building.jpg').astype(np.float32) / 255

	style_image = cv2.imread('./input_images/starry_night.jpg').astype(np.float32) / 255
	style_image_resized = cv2.resize(style_image, np.shape(content_image)[:2])

	output_dirpath = './output_images'

	neural_transfer(content_image, style_image_resized, output_dirpath)