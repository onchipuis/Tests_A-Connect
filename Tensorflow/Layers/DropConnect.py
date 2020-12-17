import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/Scripts/')
from Scripts import addMismatch

class DropConnect(tf.keras.layers.Layer):
	def __init__(self, output_size,Probability=0, isBin="no"):
		super(DropConnect, self).__init__(name='')
		self.output_size = output_size
		
		self.Probability = Probability
		
		self.isBin = isBin


	def build(self, input_shape):
		self.W = self.add_weight("kernel",
										shape = [int(input_shape[-1]),self.output_size], 
										trainable=True,
										initializer='glorot_uniform')
		self.bias = self.add_weight("bias", shape = [1 ,self.output_size], 
										trainable=True,
										initializer='zeros')
		self.dist = tfp.distributions.Bernoulli(probs=self.Probability, dtype=tf.float32)
		
		if(self.Probability != 0):
			self.Berr = self.dist.sample([1e3,1,self.output_size])
			self.Werr = self.dist.sample([1e3,int(input_shape[-1]),self.output_size])
		else:
			self.Berr = 1
			self.Werr = 1

										
	def call(self, X, training):
		self.X = X
		dim = np.shape(self.X)

		if(training):
			if(np.size(np.shape(self.X))<=3):
				batchSize = 1
				[weights,bias,Werr,Berr] = addMismatch.addMismatch(self, batchSize)	
				Z = tf.matmul(self.X, weights) + bias
			else:
				batchSize = np.shape(X)[-1]
				[weights,bias,Werr,Berr] = addMismatch.addMismatch(self,batchSize)	
				Z = tf.matmul(self.X, weights) + bias
			return Z
		
		else:
			self.X = X
			bias = self.bias
			if(self.Probability != 0):
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
			

		
		

