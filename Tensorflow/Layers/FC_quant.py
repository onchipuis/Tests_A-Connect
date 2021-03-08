import numpy as np
import tensorflow as tf

class FC_quant(tf.keras.layers.Layer):

	def __init__(self,outputSize,isBin='yes',**kwargs):
		super(FC_quant, self).__init__()
		self.outputSize = outputSize
		self.isBin = isBin


	def build(self, input_shape):
		self.W = self.add_weight("kernel",
										shape = [int(input_shape[-1]),self.outputSize], 
										trainable=True,
										initializer='glorot_uniform')
		self.bias = self.add_weight("bias", shape = [self.outputSize], 
										trainable=True,
										initializer='zeros')

	def call(self, X):
		self.X = X
		if(self.isBin=='yes'):
			weights = self.sign(self.W) #This layer is the first approach to a layer with weights binarized. Please try to ignore it. Is not tested yet, and maybe is not working ok. :)
		else:
			weights = self.W
		bias = self.bias
		Z = tf.matmul(self.X,weights) + bias
		return Z
		
	def get_config(self):
		config = super(FC_quant, self).get_config()
		config.update({
			'outputSize': self.outputSize,
			'isBin': self.isBin})
		return config		
		
	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = dy/abs(x)
			return dydx
		return y, grad





