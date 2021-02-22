import tensorflow as tf
import numpy as np
import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Scripts')
import Scripts 
from Scripts import addMismatch

#### This layer is very similar to A-Connect, please, go to AConnect.py to see the explanation of the layer.
class DropConnect(tf.keras.layers.Layer):
	def __init__(self, output_size,Wstd=0,Bstd=0,isBin="no", **kwargs):
		super(DropConnect, self).__init__()
		self.output_size = output_size
		self.Bstd = Bstd
		self.Wstd = Wstd
		self.isBin = isBin


	def build(self, input_shape):
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),self.output_size], 
										trainable=True,
										initializer='glorot_uniform')
		self.bias = self.add_weight("bias", shape = [self.output_size,], 
										trainable=True,
										initializer='zeros')
		self.dist = tfp.distributions.Bernoulli(probs=self.Wstd, dtype=tf.float32)
		
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = tf.random.stateless_binomial(shape=(self.output_size),seed=[123,456],counts=1,probs=self.Bstd) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
				self.Berr = tf.random.stateless_binomial(shape=(1000,self.output_size),seed=[123,456],counts=1,probs=self.Bstd) #"Pool" of bias error vectors
																	
			else:
				self.Berr = tf.constant(1,dtype=tf.float32)
			if(self.Wstd): 
				self.infWerr = tf.random.stateless_binomial(shape=(int(input_shape[-1]),self.output_size),seed=[123,456],counts=1,probs=self.Wstd) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				self.Werr = tf.random.stateless_binomial(shape=(1000,int(input_shape[-1]),self.output_size),seed=[123,456],counts=1,probs=self.Wstd) #"Pool" of weights error matrices.
				 
			else:
				self.Werr = tf.constant(1,dtype=tf.float32)
		else:
			self.Werr = tf.constant(1,dtype=tf.float32) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1,dtype=tf.float32)	
		self.mWerr = 1.0
		self.mBerr = 1.0
		self.membias = 1.0
		self.Xaux = 1.0	
		super(DropConnect, self).build(input_shape)				
	def call(self, X, training):
		self.X = X
		self.batch_size = tf.shape(self.X)[0]
		if(training):	
			if(self.Wstd != 0 or self.Bstd != 0):
				ID = range(np.size(self.Werr,0)) 	#This line creates a vector with numbers from 0-999 (1000 numbers)
				ID = tf.random.shuffle(ID) 			#Here is applied a shuffle or permutation of the vector numbers i.e. the output vector
													#will not have the numbers sorted from 0 to 999. Now the numbers are in random position of the vector.
													#Before the shuffle ID[0]=0, the, after the shuffle ID[0]=could be any number between 0-999.
				loc_id = tf.slice(ID, [0], [self.batch_size])	#This takes a portion of the ID vector of size batch_size. Which means if we defined 	
																	#batch_size=256. We will take only the numbers in ID in the indexes 0-255. Remeber, the numbers are sorted randomly.						   		
										   																  	
				if(self.Wstd !=0):							
					Werr = tf.gather(self.Werr,[loc_id])		#Finally, this line will take only N matrices from the "Pool" of error matrices. Where N is the batch size.          
					self.mWerr = tf.squeeze(Werr, axis=0)				#This is necessary because gather add an extra dimension. Squeeze remove this dimension.			 
																#That means, with a weights shape of [784,128] and a batch size of 256. Werr should be a tensor with shape	
																#[256,784,128], but gather return us a tensor with shape [1,256,784,128], so we remove that 1 with squeeze.
				else:
					self.mWerr = self.Werr
				if(self.isBin=='yes'):
					weights = self.sign(self.W)			#Binarize the weights and multiply them element wise with Werr mask
				else:
					weights = self.W	
				self.memW = tf.multiply(weights,Werr)			         	#Finally we multiply element-wise the error matrix with the weights.
						
				
				if(self.Bstd !=0):								#For the bias is exactly the same situation
					Berr = tf.gather(self.Berr, [loc_id])  		 
					self.mBerr = tf.squeeze(Berr,axis=0)
				else:
					self.mBerr = self.Berr
				self.membias = tf.multiply(self.mBerr,self.bias)	
				
				self.Xaux = tf.reshape(self.X, [self.batch_size,1,tf.shape(self.X)[-1]]) #We need this reshape, beacuse the input data is a column vector with
																					# 2 dimension, e.g. in the first layer using MNIST we will have a vector with
																					#shape [batchsize,784], and we need to do a matrix multiplication.
																					#Which means the last dimension of the first matrix and the first dimension of the
																					#second matrix must be equal. In this case, with a FC layer with 128 neurons we will have this dimensions
																					#[batchsize,784] for the input and for the weights mask [batchsize,784,128]
																					#And the function tf.matmul will not recognize the first dimension of X as the batchsize, so the multiplication will return a wrong result.
																					#Thats why we add an extra dimension, and transpose the vector. At the end we will have a vector with shape [batchsize,1,784].
																					#And the multiplication result will be correct.
																					
				Z = tf.matmul(self.Xaux, self.memW) 	#Matrix multiplication between input and mask. With output shape [batchsize,1,128]
				Z = tf.reshape(Z,[self.batch_size,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
				Z = tf.add(Z,self.membias) #FInally, we add the bias error mask 
				#Z = self.forward(self.W,self.bias,self.Xaux)
					
			else:
				if(self.isBin=='yes'):
					self.memW = self.sign(self.W)*self.Werr 
				else:
					self.memW = self.W*self.Werr
				bias = self.bias*self.Berr
				Z = tf.add(tf.matmul(self.X,self.memW),bias) #Custom FC layer operation when we don't have Wstd or Bstd.

		else:
		    #This part of the code will be executed during the inference
			if(self.Wstd != 0):
				Werr = self.infWerr
				Berr = self.infBerr
			else:
				Werr = self.Werr
				Berr = self.Berr
			
			if(self.isBin=='yes'):
				weights =tf.math.sign(self.W)*Werr
			else:
				weights = self.W*Werr
			bias = self.bias*Berr		
			Z = tf.add(tf.matmul(self.X, weights), bias)
					
		return Z
	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x))
			return dydx
		return y, grad		
			
	def get_config(self):
		config = super(DropConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'isBin': self.isBin
		})
		return config



		
		

