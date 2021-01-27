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

class DropConnect(tf.keras.layers.Layer):
	def __init__(self, output_size,Wstd=0, isBin="no", **kwargs):
		super(DropConnect, self).__init__()
		self.output_size = output_size
		
		self.Wstd = Wstd
		
		self.isBin = isBin


	def build(self, input_shape):
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),self.output_size], 
										trainable=True,
										initializer='glorot_uniform')
		self.bias = self.add_weight("bias", shape = [1 ,self.output_size], 
										trainable=True,
										initializer='zeros')
		self.dist = tfp.distributions.Bernoulli(probs=self.Wstd, dtype=tf.float32)
		
		if(self.Wstd != 0):
			self.Berr = self.dist.sample([1e3,1,self.output_size])
			self.Werr = self.dist.sample([1e3,int(input_shape[-1]),self.output_size])
			self.Werr = self.Werr.numpy()
			self.Berr = self.Berr.numpy()
		else:
			self.Berr = 1
			self.Werr = 1

										
	def call(self, X, training):
		self.X = X
		dim = np.shape(self.X)

		if(training):
			if(self.Wstd != 0):
				for j in range(int(len(self.X)/np.shape(X)[0])):
					if(np.size(np.shape(self.X))<2):
						batchSize = 1
					else:
						batchSize = np.shape(X)[0]
					for i in range(batchSize):
						[weights,bias,Werr,Berr] = addMismatch.addMismatch(self, batchSize)	
						Z = tf.matmul(self.X[(j)*batchSize:(j+1)*(batchSize),:],weights) + bias
			else:
				weights = self.W
				bias = self.bias
				Z = tf.matmul(self.X,weights) + bias
		
		else:
			self.X = X
			bias = self.bias
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:,:]
			else:
				Werr = self.Werr
				Berr = self.Berr
			
			if self.isBin == "yes":
				weights = tf.math.sign(self.W)
			else:
				weights= self.W
			
			weights = weights*Werr
			bias = bias*Berr
			
			Z = tf.matmul(self.X, weights) + bias
		return Z
			
	def get_config(self):
		config = super(DropConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'isBin': self.isBin
		})
		return config



		
		

