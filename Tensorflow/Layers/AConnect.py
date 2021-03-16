### Modified by: Ricardo Vergel - Dec 03/2020
###
### A-Connect fully connected layer

import tensorflow as tf
import numpy as np
############ This layer was made using the template provided by Keras. For more info, go to the official site.
"""
Fully Connected layer with A-Connect
INPUT ARGUMENTS:
-output_size: Number of neurons that you want in the layer
-Wstd and Bstd: Weights and bias standard deviation
-isBin: WHenever you want binary weights
-pool: NUmber of error matrices for training. The recomended size is the same as the batch. 
"""

class AConnect(tf.keras.layers.Layer): 
	def __init__(self,output_size,Wstd=0,Bstd=0,isBin = "no",pool=1000,d_type=tf.dtypes.float32,**kwargs): #__init__ method is the first method used for an object in python to initialize the ...
		super(AConnect, self).__init__()						 		#...object attributes
		self.output_size = output_size							 		#output_size is the number of neurons of the layer
		self.Wstd = Wstd										 		#Wstd standard deviation of the weights(number between 0-1. By default is 0)
		self.Bstd = Bstd										 		#Bstd standard deviation of the bias(number between 0-1. By default is 0)
		self.isBin = isBin                                       		#if the layer will binarize the weights(String yes or no. By default is no)
		self.pool = pool                                                  #Multiplier for the pool of error matrices, by default is 1
		self.d_type = d_type
	def build(self,input_shape):								 #This method is used for initialize the layer variables that depend on input_shape
								                                    #input_shape is automatically computed by tensorflow
		self.W = self.add_weight("W",							
										shape = [int(input_shape[-1]),self.output_size], #Weights matrix 
										initializer = "glorot_uniform",
										trainable=True)

		self.bias = self.add_weight("bias",
										shape = [self.output_size,],					#Bias vector
										initializer = "zeros",
										trainable=True)					
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = abs(1+tf.random.normal(shape=[self.output_size],stddev=self.Bstd)) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
				self.Berr = abs(1+tf.random.normal(shape=[self.pool,self.output_size],stddev=self.Bstd,dtype=self.d_type)) #"Pool" of bias error vectors
																	
			else:
				self.Berr = tf.constant(1,dtype=tf.float32)
			if(self.Wstd): 
				self.infWerr = abs(1+tf.random.normal(shape=[int(input_shape[-1]),self.output_size],stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				self.Werr = abs(1+tf.random.normal(shape=[self.pool,int(input_shape[-1]),self.output_size],stddev=self.Wstd,dtype=self.d_type)) #"Pool" of weights error matrices.
				 
			else:
				self.Werr = tf.constant(1,dtype=tf.float32)
		else:
			self.Werr = tf.constant(1,dtype=tf.float32) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1,dtype=tf.float32)
		super(AConnect, self).build(input_shape)
		
	def call(self, X, training=None): #With call we can define all the operations that the layer do in the forward propagation.
		self.X = X
		self.batch_size = tf.shape(self.X)[0] #Numpy arrays and tensors have the number of array/tensor in the first dimension.
											  #i.e. a tensor with this shape [1000,784,128] are 1000 matrix of [784,128].
											  #Then the batch_size of the input data also is the first dimension. 
											  
		#This code will train the network. For inference, please go to the else part
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
				self.memW = tf.multiply(weights,self.mWerr)			         	#Finally we multiply element-wise the error matrix with the weights.
						
				
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
		
	#THis is only for saving purposes. Does not affect the layer performance.	
	def get_config(self):
		config = super(AConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'Bstd': self.Bstd,
			'isBin': self.isBin,
            'pool': self.pool,
            'd_type': self.d_type})
		return config
		
	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x))
			return dydx
		return y, grad
###HOW TO IMPLEMENT MANUALLY THE BACKPROPAGATION###		
	"""
	@tf.custom_gradient
	def forward(self,W,bias,X):
		ID = range(np.size(self.Werr,0))			#Generate and shuffle a vector of 1000 elements between 0.999
		ID = tf.random.shuffle(ID)  
		loc_id = tf.slice(ID, [0], [self.batch_size]) #Take a portion of size batch_size from ID
		if(self.Wstd !=0):							
			Werr = tf.gather(self.Werr,[loc_id])		#Finally, this line will take only N matrices from the "Pool" of error matrices. Where N is the batch size.          
			self.mWerr = tf.squeeze(Werr, axis=0)		
		else:
			self.mWerr = self.Werr
		if(self.isBin=='yes'):
			weights = tf.math.sign(W)			#Binarize the weights
		else:
			weights = W
		if(self.Bstd !=0):								#For the bias is exactly the same situation
			Berr = tf.gather(self.Berr, [loc_id])  		 
			self.mBerr = tf.squeeze(Berr,axis=0)
		else:
			self.mBerr = self.Berr
		weights = tf.reshape(weights, [1,tf.shape(weights)[0],tf.shape(weights)[1]])
		loc_W = weights*self.mWerr 				#Get the weights with the error matrix included. Also takes the binarization error when isBin=yes
		bias = tf.reshape(bias, [1,tf.shape(bias)[0]])
		loc_bias = bias*self.mBerr
		Z = tf.matmul(X,loc_W)
		Z = tf.reshape(Z, [self.batch_size,tf.shape(Z)[-1]]) #Reshape Z to column vector
		Z = tf.add(Z, loc_bias) # Add the bias error mask
		def grad(dy):
			if(self.isBin=="yes"):
				layerW = tf.reshape(W, [1,tf.shape(W)[0],tf.shape(W)[1]])
				Werr = loc_W/layerW		#If the layer is binary we use Werr as W*/layer.W as algorithm 3 describes.
			else:
				Werr = self.mWerr  #If not, Werr will be the same matrices that we multiplied before
			dy = tf.reshape(dy, [self.batch_size,1,tf.shape(dy)[-1]]) #Reshape dy to [batch,1,outputsize]
			dX = tf.matmul(dy,loc_W, transpose_b=True) #Activations gradient
			dX = tf.reshape(dX, [self.batch_size, tf.shape(dX)[-1]])
			dWerr = tf.matmul(X,dy,transpose_a=True) #Gradient for the error mask of weights
			dBerr = tf.reshape(dy, [self.batch_size,tf.shape(dy)[-1] ]) #Get the gradient of the error mask of bias with property shape
			dW = dWerr*Werr #Get the property weights gradient
			dW = tf.reduce_sum(dW, axis=0)
			dB = dBerr*self.mBerr #Get the property bias gradient
			dB = tf.reduce_sum(dB, axis=0)
			return dW,dB,dX
		return Z, grad """
			

