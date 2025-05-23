import numpy as np 
import tensorflow as tf 


from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, metrics

class SiameseNetwork(Model):

	def __init__(self, model, alpha= 0.2):
		super(SiameseNetwork, self).__init__()
		self.model = model 
		self.alpha = alpha
		self.loss_tracker = metrics.Mean(name='loss')
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00006)


	@tf.function
	def call(self, inputs):
		#return tf.math.l2_normalize(self.model(inputs), axis=-1)

		image_1, image_2, image_3 =  inputs
		with tf.name_scope("Anchor") as scope:
			feature_1 = self.model(image_1)
			feature_1 = tf.math.l2_normalize(feature_1, axis=-1)
		with tf.name_scope("Positive") as scope:
			feature_2 = self.model(image_2)
			feature_2 = tf.math.l2_normalize(feature_2, axis=-1)
		with tf.name_scope("Negative") as scope:
			feature_3 = self.model(image_3)
			feature_3 = tf.math.l2_normalize(feature_3, axis=-1)
		return [feature_1, feature_2, feature_3]


	def train_step(self, data):

		with tf.GradientTape() as tape:
			y = self.call(data)
			loss = self.triplet_loss(y)

		gradients = tape.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_weights)
			)
		self.loss_tracker.update_state(loss)

		return {'loss': self.loss_tracker.result()}

	def triplet_loss(self, x):
		anchor, positive, negative =  x 
		pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
		neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
		basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.alpha)
		loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
		return loss


	@tf.function
	def get_features(self, inputs):
		return tf.math.l2_normalize(self.model(inputs), axis=-1)


	@property
	def metrics(self):
		return [self.loss_tracker]
	

