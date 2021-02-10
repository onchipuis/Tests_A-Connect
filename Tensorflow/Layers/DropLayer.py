### Modified by: Ricardo M. Vergel S. - October 28/2020
### Custom Dropout layer

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class DropLayer(tf.keras.layers.Layer):
	
	def __init__(self, drop_ratio):
		super(DropLayer,self).__init__() 
		self.drop_ratio = drop_ratio
		self.mask = []

	def build(self, input_shape):

		self.dist = tfp.distributions.Bernoulli(probs=self.drop_ratio, dtype=tf.float32) #Create a bernoulli distribution to create the mask

		
	
	def call(self, X, training):
		self.X = X
		inputshape= np.shape(self.X)
		if(training):
			self.mask = self.dist.sample([inputshape[0],inputshape[-1]])/self.drop_ratio #Get the mask and apply the dropout formula
			Xrand = tf.math.multiply(self.X,self.mask) #Get the output with activations dropped.
			return tf.reshape(Xrand, tf.shape(self.X))
		else:
			return X
		

	#def backward(self,output_error, learning_rate):
	#	input_error = tf.matmul(output_error, self.mask)
	#	return input_error	


