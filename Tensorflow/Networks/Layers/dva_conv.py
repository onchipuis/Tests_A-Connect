import tensorflow as tf
import numpy as np


class dva_conv(tf.keras.layers.Layer):
	def __init__(self,filters,kernel_size,strides=1,padding="VALID",Wstd=0,Bstd=0,isBin='no',weights_regularizer=None,bias_regularizer=None,**kwargs):
		super(dva_conv, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.Wstd = Wstd
		self.Bstd = Bstd    
		self.isBin = isBin
		self.strides = strides
		self.padding = padding
		self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None               
	def build(self,input_shape):
		self.shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB

		self.W = self.add_weight('kernel',
								  shape = self.shape,
								  initializer = "glorot_uniform",
                                  regularizer = self.weights_regularizer,
								  trainable=True)				  
		self.bias = self.add_weight('bias',
									shape=(self.filters,),
									initializer = 'zeros',
                                    regularizer = self.bias_regularizer,
									trainable=True)
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = abs(1+tf.random.normal(shape=[self.filters,],stddev=self.Bstd)) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
																	
			else:
				self.Berr = tf.constant(1)
			if(self.Wstd !=0): 
				self.infWerr = abs(1+tf.random.normal(shape=self.shape,stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				 
			else:
				self.Werr = tf.constant(1)
		else:
			self.Werr = tf.constant(1) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1)
		super(dva_conv, self).build(input_shape)
		
	def call(self, X, training=None): #With call we can define all the operations that the layer do in the forward propagation.
		self.X = tf.cast(X, dtype=tf.dtypes.float32)     
		self.batch_size = tf.shape(self.X)[0]     
		if training:
		    if self.isBin == "yes":
		        W = self.sign(self.W)
		    else:
		        W = self.W            
		    Werr = abs(1+tf.random.normal(shape=self.shape,stddev=self.Wstd))
		    Berr = abs(1+tf.random.normal(shape=[self.filters,],stddev=self.Bstd))
		    weights = W*Werr
		    bias = self.bias*Berr
		    Z = tf.nn.conv2d(self.X, weights,strides=[1,self.strides,self.strides,1],padding=self.padding) + bias
		else:
		    Werr = self.infWerr
		    Berr = self.infBerr
		    if(self.isBin=='yes'):
		        weights =tf.math.sign(self.W)*Werr
		    else:
		        weights = self.W*Werr   
		    bias = self.bias*Berr
		    Z = tf.nn.conv2d(self.X, weights,strides=[1,self.strides,self.strides,1],padding=self.padding) + bias
		return Z

	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x))
			return dydx
		return y, grad     

	def get_config(self):
		config = super(dva_conv, self).get_config()
		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'Wstd': self.Wstd,
			'Bstd': self.Bstd,
			'isBin': self.isBin,
			'strides': self.strides,
			'padding': self.padding})
		return config
