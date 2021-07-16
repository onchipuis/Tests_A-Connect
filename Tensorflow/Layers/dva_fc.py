import tensorflow as tf
import numpy as np

class dva_fc(tf.keras.layers.Layer):
	def __init__(self,output_size,Wstd=0,Bstd=0,isBin="no",weights_regularizer=None,bias_regularizer=None
    ,**kwargs): #__init__ method is the first method used for an object in python to initialize the ...
		super(dva_fc, self).__init__()						 		#...object attributes
		self.output_size = output_size							 		#output_size is the number of neurons of the layer
		self.Wstd = Wstd										 		#Wstd standard deviation of the weights(number between 0-1. By default is 0)
		self.Bstd = Bstd										 		#Bstd standard deviation of the bias(number between 0-1. By default is 0)       
		self.isBin = isBin                                       		#if the layer will binarize the weights(String yes or no. By default is no)
                                         #Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
		self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
	def build(self,input_shape):								 #This method is used for initialize the layer variables that depend on input_shape
								                                    #input_shape is automatically computed by tensorflow
		self.W = self.add_weight("W",							
										shape = [int(input_shape[-1]),self.output_size], #Weights matrix 
										initializer = "glorot_uniform",
                                        regularizer = self.weights_regularizer,
										trainable=True)

		self.bias = self.add_weight("bias",
										shape = [self.output_size,],					#Bias vector
										initializer = "zeros",
                                        regularizer = self.bias_regularizer,
										trainable=True)					
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = abs(1+tf.random.normal(shape=[self.output_size],stddev=self.Bstd)) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
				
																	
			else:
				self.Berr = tf.constant(1)
			if(self.Wstd != 0): 
				self.infWerr = abs(1+tf.random.normal(shape=[int(input_shape[-1]),self.output_size],stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				
				 
			else:
				self.Werr = tf.constant(1)
		else:
			self.Werr = tf.constant(1) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1)
		super(dva_fc, self).build(input_shape)
		
	def call(self, X, training=None): #With call we can define all the operations that the layer do in the forward propagation.
		self.X = tf.cast(X, dtype=tf.dtypes.float32)
		row = tf.shape(self.X)[-1]         
		if training:   
		    if self.isBin == "yes":
		        W = self.sign(self.W)
		    else:
		        W = self.W 
		    Werr = abs(1+tf.random.normal(shape=[tf.cast(row,tf.int32),self.output_size],stddev=self.Wstd))
		    Berr = abs(1+tf.random.normal(shape=[self.output_size,],stddev=self.Bstd))
		    weights = W*Werr
		    bias = self.bias*Berr
		    Z = tf.matmul(self.X,weights)+bias
		else:
		    Werr = self.infWerr
		    Berr = self.infBerr
		    if(self.isBin=='yes'):
		        weights =tf.math.sign(self.W)*Werr
		    else:
		        weights = self.W*Werr            
		    bias = self.bias*Berr
		    Z = tf.matmul(self.X,weights)+bias		
		return Z

	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x))
			return dydx
		return y, grad        

	def get_config(self):
			config = super(dva_fc, self).get_config()
			config.update({
				'output_size': self.output_size,
				'Wstd': self.Wstd,
				'Bstd': self.Bstd,
				'isBin': self.isBin,
		        'weights_regularizer': self.weights_regularizer,
		        'bias_regularizer' : self.bias_regularizer})
			return config

