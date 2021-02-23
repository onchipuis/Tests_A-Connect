import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

class Conv(tf.keras.layers.Layer):

	def __init__(self,filters,kernel_size,strides=(1,1),padding='same',**kwargs):
		super(Conv, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding	
		
	def build(self,input_shape):
		shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB
		self.W = self.add_weight('kernel',
								  shape = shape,
								  initializer = "glorot_uniform",
								  trainable=True)				  
		self.bias = self.add_weight('bias',
									shape=(self.filters,),
									initializer = 'zeros',
									trainable=True)
		if self.padding == 'same' or "same":
			self.padding = "SAME"
		elif self.padding == 'valid' or "valid":
			self.padding = "VALID"
		super(Conv, self).build(input_shape)
	def call(self,X,training):
		self.X = float(X)
		Z = tf.nn.convolution(self.X,self.W,self.strides,self.padding)
		Z = self.bias+Z
		return Z
		
	def get_config(self):
		config = super(Conv, self).get_config()
		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'padding': self.padding})
		return config
	
