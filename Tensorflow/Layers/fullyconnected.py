### Modified by : Ricardo Vergel - October 17/2020
### November 03/2020: Backpropagation method doesnt work because fit() method will never call it
### 
### Custom Fully connected layer

import numpy as np
import tensorflow as tf

#Clase fullyconnected

class fullyconnected(tf.keras.layers.Layer):
	def __init__(self, n_neurons):
		super(fullyconnected, self).__init__()
		self.n_neurons = n_neurons
	
	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
										shape = [int(input_shape[-1]),
												self.n_neurons],
										#initializer = "random_normal",
										trainable=True)
		self.bias = self.add_weight("bias",
										shape = [1,
												self.n_neurons],
										#initializer = "random_normal",
										trainable=True)

##Feedforward
	def call(self, X):
		self.X = X
		return tf.matmul(self.X, self.kernel) + self.bias

##Backpropagation




