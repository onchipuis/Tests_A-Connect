import numpy as np
import tensorflow as tf

class ConvAConnect(tf.keras.layers.Layer):
	def __init__(self,filters,kernel_size,Wstd=0,Bstd=0,isBin='no',strides=1,padding="SAME",activation=None,**kwargs):
		super(ConvAConnect, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.Wstd = Wstd
		self.Bstd = Bstd
		self.isBin = isBin
		self.strides = strides
		self.padding = padding
		self.activation = tf.keras.activations.get(activation) 	
		
	def build(self,input_shape):
		shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB

		self.W = self.add_weight('kernel',
								  shape = shape,
								  initializer = "glorot_uniform",
								  trainable=True)				  
		self.bias = self.add_weight('bias',
									shape=(self.filters,),
									initializer = 'zeros',
									trainable=True)
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = abs(1+tf.random.normal(shape=[self.filters,],stddev=self.Bstd)) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
				self.Berr = abs(1+tf.random.normal(shape=[1000,self.filters],stddev=self.Bstd)) #"Pool" of bias error vectors
																	
			else:
				self.Berr = tf.constant(1,dtype=tf.float32)
			if(self.Wstd): 
				self.infWerr = abs(1+tf.random.normal(shape=shape,stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				self.Werr = abs(1+tf.random.normal(shape=list((1000,))+shape,stddev=self.Wstd)) #"Pool" of weights error matrices. Here I need to add an extra dimension. So I concatenate it. But to concatenate, the two elements must be the same type, in this cases, the two elements must be a list
				#self.Werr = tf.squeeze(self.Werr, axis=0) # Remove the extra dimension
				 
			else:
				self.Werr = tf.constant(1,dtype=tf.float32)
		else:
			self.Werr = tf.constant(1,dtype=tf.float32) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1,dtype=tf.float32)
		super(ConvAConnect, self).build(input_shape)
	def call(self,X,training):
		self.X = X
		self.batch_size = tf.shape(self.X)[0]
		if(training):
			if(self.Wstd != 0 or self.Bstd != 0):
				ID = range(np.size(self.Werr,0))
				ID = tf.random.shuffle(ID)
				
				loc_id = tf.slice(ID,[0],[self.batch_size])
				
				if(self.Wstd != 0):
					Werr = tf.gather(self.Werr,[loc_id])
					Werr = tf.squeeze(Werr, axis=0)
				else:
					Werr = self.Werr
				if(self.isBin=='yes'):
					weights=self.sign(self.W)
				else:
					weights=self.W
				weights = tf.expand_dims(weights,axis=0)
				memW = tf.multiply(weights,Werr)
				if(self.Bstd != 0):
					Berr = tf.gather(self.Berr, [loc_id])
					Berr = tf.squeeze(Berr, axis=0)
				else:
					Berr = self.Berr
				bias = tf.expand_dims(self.bias,axis=0)
				membias = tf.multiply(bias,Berr)                
				membias = tf.reshape(membias,[self.batch_size,1,1,tf.shape(membias)[-1]])
				Xaux = self.X#tf.reshape(self.X, [self.batch_size,tf.shape(self.X)[1],tf.shape(self.X)[2],tf.shape(self.X)[3]])
				Z = tf.squeeze(tf.map_fn(self.conv,(tf.expand_dims(Xaux,1),memW),dtype=tf.float32),axis=1)#tf.nn.convolution(Xaux,memW,self.strides,self.padding)
				#Z = tf.reshape(Z, [self.batch_size, tf.shape(Z)[2],tf.shape(Z)[3],tf.shape(Z)[4]])
				Z = membias+Z
				if(self.activation is not None):
					Z=self.activation(Z)					
			else:
				if(self.isBin=='yes'):
					weights=self.sign(self.W)*self.Werr
				else:
					weights=self.W*self.Werr
				Z = self.bias*self.Berr+tf.nn.convolution(self.X,weights,self.strides,self.padding)
				if(self.activation is not None):
					Z=self.activation(Z)					
		else:
			if(self.Wstd != 0 or self.Bstd !=0):
				if(self.Wstd !=0):
					Werr = self.infWerr
				else:
					Werr = self.Werr
				if(self.Bstd != 0):
					Berr = self.infBerr
				else:
					Berr = self.Berr
			else:
				Werr = self.Werr
				Berr = self.Berr		
			if(self.isBin=='yes'):
				weights=tf.math.sign(self.W)*Werr
			else:
				weights=self.W*Werr	
			bias = self.bias*Berr                
			Z = bias+tf.nn.convolution(self.X,weights,self.strides,self.padding)	
			if(self.activation is not None):
				Z=self.activation(Z)            								
		return Z
	def conv(self,tupla):
		x,kernel = tupla
		return tf.nn.convolution(x,kernel,strides=self.strides,padding=self.padding)    
	def get_config(self):
		config = super(ConvAConnect, self).get_config()
		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'Wstd': self.Wstd,
			'Bstd': self.Bstd,
			'isBin': self.isBin,
			'strides': self.strides,
			'padding': self.padding,
			'activation':self.activation})
		return config
	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x))
			return dydx
		return y, grad		
