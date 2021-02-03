### Modified by: Ricardo Vergel - Dec 03/2020
###
### A-Connect fully connected layer

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Scripts')
import Scripts 
from Scripts import addMismatch

class AConnect(tf.keras.layers.Layer):
	def __init__(self,output_size,Wstd=0,isBin = "no",**kwargs):
		super(AConnect, self).__init__()
		self.output_size = output_size
		self.Wstd = Wstd
		self.isBin = isBin
		
	def build(self,input_shape):
		self.batch_size = input_shape[0]
	
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),self.output_size],
										initializer = "glorot_uniform",
										trainable= True)
		self.bias = self.add_weight("bias",
										shape = [1,self.output_size],
										initializer = "zeros",
										trainable= True)
		if(self.Wstd != 0):
			self.Berr = abs(np.random.normal(scale=self.Wstd,size=[1000,1,self.output_size]))
			self.Berr = self.Berr.astype('float32')
			self.Werr = abs(np.random.normal(scale=self.Wstd,size=[1000,int(input_shape[-1]),self.output_size]))
			self.Werr = self.Werr.astype('float32')

		else:
			self.Werr = 1
			self.Berr = 1
		
	def call(self, X, training):
		self.X = X
	
		if(training):
			if(self.Wstd != 0):
				self.batch_size = tf.shape(self.X)[0]
				self.X = tf.reshape(self.X, [self.batch_size,1, np.shape(self.X)[1]])
				[self.memweights, self.membias, self.memWerr, self.memBerr] = addMismatch.addMismatch(self, self.batch_size)
				Z = tf.matmul(self.X,self.memweights)
				Z = tf.add(Z, self.membias)
			else:
				weights = self.W
				bias = self.bias
				Z = tf.matmul(self.X,self.W)
				Z = tf.add(Z,self.bias)



		else:
		
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:,:]
			else:
				Werr = self.Werr
				Berr = self.Berr
			weights = tf.multiply(self.W,Werr)
			bias = tf.multiply(self.bias,Berr)
		#	for i in range(self.batch_size):
			Z = tf.matmul(self.X, weights)
			Z = tf.add(Z, bias)

			
		return Z
		
	def get_config(self):
		config = super(AConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'isBin': self.isBin})
		return config
		

