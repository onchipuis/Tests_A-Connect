### Modified by: Ricardo Vergel - Dec 03/2020
###
### First approach to an A-Connect fully connected layer

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/Scripts/')

class AConnect(tf.keras.layers.Layer):
	def __init__(self,output_size,Wstd=0, **kwargs):
		super(AConnect, self).__init__()
		self.output_size = output_size
		self.Wstd = Wstd
		
	def build(self,input_shape):
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),self.output_size],
										initializer = "glorot_uniform",
										trainable= True)
		self.bias = self.add_weight("bias",
										shape = [1,self.output_size],
										initializer = "zeros",
										trainable= True)
		if(self.Wstd != 0):
			self.dist = tfp.distributions.Normal(loc=1,scale=self.Wstd)
			self.Berr = self.dist.sample([1e3,1,self.output_size])
			self.Werr = self.dist.sample([1e3,int(input_shape[-1]),self.output_size])
			self.Werr = self.Werr.numpy()
			self.Berr = self.Berr.numpy()
		else:
			self.Werr = 1
			self.Berr = 1
		
		
	def call(self, X, training):
		self.X = X
		if(training):
			if(self.Wstd != 0):
				#if(np.size(np.shape(self.X))<=3):
				#	batchSize = 1
				#else:
				#	batchSize = np.shape(X)[-1]

				dim = np.shape(self.Werr)
				ID = np.random.randint(0,dim[0]-1)
				Werr = self.Werr[ID,:,:]
				Berr = self.Berr[ID,:,:]
				weights = self.W*Werr
				bias = self.bias*Berr
				Z = tf.matmul(self.X,weights) + bias
			else:
				weights = self.W
				bias = self.bias
				Z = tf.matmul(self.X,weights) + bias
		else:
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:,:]
			else:
				Werr = self.Werr
				Berr = self.Berr
			weights = self.W*Werr
			bias = self.bias*Berr
			Z = tf.matmul(self.X,weights) + bias
		return Z
		
	def get_config(self):
		config = super(AConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd})
		return config
		

