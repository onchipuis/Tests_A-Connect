import numpy as np
import tensorflow as tf

class FC_quant(tf.keras.layers.Layer):

	def __init__(self,output_shape):
		super(FC_quant, self).__init__()
		self.outputSize = output_shape
		self.isBin = "yes"
		self.Werr = 1
		self.Berr = 1

	def build(self, input_shape):
		self.W = self.add_weight("kernel",
										shape = [int(input_shape[-1]),self.outputSize], 
										trainable=True,
										initializer='glorot_uniform')
		self.bias = self.add_weight("bias", shape = [1 ,self.outputSize], 
										trainable=True,
										initializer='zeros')

	def call(self, X):
		Werr = self.Werr
		Berr = self.Berr 
		self.X = X
		W = tf.math.sign(self.W)*Werr #This layer is the first approach to a layer with weights binarized. Please try to ignore it. Is not tested yet, and maybe is not working ok. :)
		
		bias = tf.math.sign(self.bias)*Berr
		
		Z = tf.matmul(self.X, W) + bias
		return Z
		

