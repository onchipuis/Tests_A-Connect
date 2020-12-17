import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/Scripts/')
from Scripts import addMismatch
from Scripts import Xreshape
		

class dropconnect2(tf.keras.layers.Layer):
	def __init__(self, output_size,Probability=0, isBin="no"):
		super(dropconnect2, self).__init__(name='')
		self.output_size = output_size
		
		self.Probability = Probability
		
		self.isBin = isBin
		
		self.memory = [[],[],[],[]]


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
		[batchSize, Xsz] = Xreshape.Xreshape(self.X)
		if(training):
			[weights,bias,Werr,Berr] = addMismatch.addMismatch(self,batchSize)
			self.memory[0] = weights
			self.memory[1] = bias
			self.memory[2] = Werr
			self.memory[3] = Berr
			Z = tf.matmul(self.X, self.memory[0]) + self.memory[1]
			return Z
		
		else:
			self.X = X
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
			bias = self.bias*Berr
			
			Z = tf.matmul(self.X, weights) + bias
			return Z
			
	def backward(self,dLdZ,learning_rate):
		[batchSize, Xsz] = Xreshape.Xreshape(self.X)
		Werr = self.memory[2]
		weights = self.memory[0]
		
		Berr = self.memory[3]
		bias = self.bias
		
		dLdZaux = tf.reshape(dLdZ,[],1,batchSize)
		dLdXaux = tf.matmul(dLdZaux, weights.T)
		dLdX = np.reshape(dLdXaux, [Xsz, batchSize])
		dLdWaux = tf.matmul(self.X.T,dLdZaux)
		dLdBaux = tf.matmul(dLdZaux,Berr) 
            
		dLdW = sum(tf.matmul(dLdWaux,Werr),3)
		dLdB = sum(dLdBaux,3)
		dLdB = np.reshape(dLdB,[],1)
		


