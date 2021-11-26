### Modified by : Ricardo Vergel - October 17/2020
### November 03/2020: Backpropagation method doesnt work because fit() method will never call it
### 
### Custom Fully connected layer

import numpy as np
import tensorflow as tf

#Clase fullyconnected

# There is nothing to explain here, is a custom fully connected layer with the template provided by keras

class fullyconnected(tf.keras.layers.Layer):
	def __init__(self, n_neurons, **kwargs):
		super(fullyconnected, self).__init__()
		self.n_neurons = n_neurons
	
	def build(self, input_shape):
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),
												self.n_neurons],
										initializer = "glorot_uniform",
										trainable=True)
		self.bias = self.add_weight("bias",
										shape = [self.n_neurons],
										initializer = "zeros",
										trainable=True)

##Feedforward
	def call(self, X):
		self.X = X
		self.Zaux = tf.matmul(self.X, self.W)
		Z = tf.add(self.Zaux,self.bias)
#		tf.print(tf.shape(X))
		
		return Z

	def get_config(self):
		config = super(fullyconnected, self).get_config()
		config.update({
			'n_neurons': self.n_neurons,
			})
		return config
	

		
			



