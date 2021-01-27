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
				for j in range(int(len(self.X)/np.shape(X)[0])):
					if(np.size(np.shape(self.X))<2):
						batchSize = 1
					else:
						batchSize = np.shape(X)[0]
					#print(batchSize)
					for i in range(batchSize):
						dim = np.shape(self.Werr)
						self.ID = np.random.randint(0,dim[0]-1)
						Werr = self.Werr[self.ID,:,:]
						#print(self.ID)
						Berr = self.Berr[self.ID,:,:]
						weights = self.W*Werr
						bias = self.bias*Berr
						Z = tf.matmul(self.X[(j)*batchSize:(j+1)*(batchSize),:],weights) + bias
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
		

