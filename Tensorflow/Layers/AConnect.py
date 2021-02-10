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
############ This layer was made using the template provided by Keras. For more info, go to the official site.

class AConnect(tf.keras.layers.Layer): 
	def __init__(self,output_size,Wstd=0,isBin = "no",**kwargs): #__init__ method is the first method used for an object in python to initialize the ...
		super(AConnect, self).__init__()						 #...object attributes
		self.output_size = output_size							 #output_size is the number of neurons of the layer
		self.Wstd = Wstd										 #Wstd standard deviation of the weights(number between 0-1. By default is 0)
		self.isBin = isBin                                       #if the layer will binarize the weights(String yes or no. By default is no)
		
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
		if(self.Wstd != 0): #If the layer will take into account the standard deviation of the weights 
			self.Berr = tf.random.normal(shape=[1000,self.output_size],stddev=self.Wstd) #"Pool" of bias error vectors
			self.Berr = abs(1+self.Berr.numpy()) #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
												 #This was necessary to change the error matrix/array when Monte Carlo was running.
			self.Werr = tf.random.normal(shape=[1000,int(input_shape[-1]),self.output_size],stddev=self.Wstd) #"Pool" of weights error matrices.
			self.Werr = abs(1+self.Werr.numpy()) 
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
			if(self.Wstd != 0):
				ID = range(np.size(self.Werr,0)) #This line creates a vector with numbers from 0-999 (1000 numbers)
				ID = tf.random.shuffle(ID) #Here is applied a shuffle or permutation of the vector numbers i.e. the output vector
										   #will not have the numbers sorted from 0 to 999. Now the numbers are in random position of the vector.
										   #Before the shuffle ID[0]=0, the, after the shuffle ID[0]=could be any number between 0-999.
										   
				loc_id = tf.slice(ID, [0], [self.batch_size]) #This takes a portion of the ID vector of size batch_size. Which means if we defined
															  #batch_size=256. We will take only the numbers in ID in the indexes 0-255. Remeber, the numbers
															  #are sorted randomly.
				Werr = tf.gather(self.Werr,[loc_id])          #Finally, this line will take only N matrices from the "Pool" of error matrices. Where N is the batch size.
				Werr = tf.squeeze(Werr, axis=0)				  #This is necessary because gather add an extra dimension. Squeeze remove this dimension.
															  #That means, with a weights shape of [784,128] and a batch size of 256. Werr should be a tensor with shape
															  #[256,784,128], but gather return us a tensor with shape [1,256,784,128], so we remove that 1 with squeeze.
				self.memW = tf.multiply(self.W,Werr)          #Finally we multiply element-wise the error matrix with the weights.
				Berr = tf.gather(self.Berr, [loc_id])  		  #For the bias is exactly the same situation
				Berr = tf.squeeze(Berr,axis=0)
				self.membias = tf.multiply(self.bias,Berr)
				
				Xaux = tf.reshape(self.X, [self.batch_size,1,tf.shape(self.X)[-1]]) #We need this reshape, beacuse the input data is a column vector with
																					# 2 dimension, e.g. in the first layer using MNIST we will have a vector with
																					#shape [batchsize,784], and we need to do a matrix multiplication.
																					#Which means the last dimension of the first matrix and the first dimension of the
																					#second matrix must be equal. In this case, with a FC layer with 128 neurons we will have this dimensions
																					#[batchsize,784] for the input and for the weights mask [batchsize,784,128]
																					#And the function tf.matmul will not recognize the first dimension of X as the batchsize, so the multiplication will return a wrong result.
																					#Thats why we add an extra dimension, and transpose the vector. At the end we will have a vector with shape [batchsize,1,784].
																					#And the multiplication result will be correct.
				Z = tf.matmul(Xaux, self.memW) 	#Matrix multiplication between input and mask. With output shape [batchsize,1,128]
				Z = tf.reshape(Z,[self.batch_size,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
				Z = tf.add(Z,self.membias) #FInally, we add the mask error of bias.
			
		
			else:
				Z = tf.add(tf.matmul(self.X,self.W),self.bias) #Custom FC layer operation when we don't have Wstd.

		else:
		    #This part of the code will be executed during the inference
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:]
				
			else:
				Werr = self.Werr
				Berr = self.Berr
				
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
			'isBin': self.isBin})
		return config

