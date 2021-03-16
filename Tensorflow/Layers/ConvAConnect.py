import numpy as np
import tensorflow as tf
"""
Convolutional layer with A-Connect
INPUT ARGUMENTS:
-filters: Number of filter that you want to use during the convolution.(Also known as output channels)
-kernel_size: List with the dimension of the filter. e.g. [3,3]. It must be less than the input data size
-Wstd and Bstd: Weights and bias standard deviation for training
-isBin: string yes or no, whenever you want binary weights
-strides: Number of strides (or steps) that the filter moves during the convolution
-padding: "SAME" or "VALID". If you want to keep the same size or reduce it.
-pool: NUmber of error matrices for training. The recomended size is the same as the batch. 
"""
class ConvAConnect(tf.keras.layers.Layer):
	def __init__(self,filters,kernel_size,Wstd=0,Bstd=0,isBin='no',strides=1,padding="VALID",pool=1000,d_type=tf.dtypes.float32,**kwargs):
		super(ConvAConnect, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.Wstd = Wstd
		self.Bstd = Bstd
		self.isBin = isBin
		self.strides = strides
		self.padding = padding
		self.pool = pool
		self.d_type = d_type
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
				self.Berr = abs(1+tf.random.normal(shape=[self.pool,self.filters],stddev=self.Bstd,dtype=self.d_type)) #"Pool" of bias error vectors
																	
			else:
				self.Berr = tf.constant(1,dtype=tf.float32)
			if(self.Wstd): 
				self.infWerr = abs(1+tf.random.normal(shape=shape,stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				self.Werr = abs(1+tf.random.normal(shape=list((self.pool,))+shape,stddev=self.Wstd,dtype=self.d_type)) #"Pool" of weights error matrices. Here I need to add an extra dimension. So I concatenate it. But to concatenate, the two elements must be the same type, in this cases, the two elements must be a list
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
																#All this code works exactly as A-Connect fullyconnected layer.
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
				#################################WORST OPTION TO DO THE CONVOLUTION###############################################################
				#Xaux = self.X#tf.reshape(self.X, [self.batch_size,tf.shape(self.X)[1],tf.shape(self.X)[2],tf.shape(self.X)[3]])
				#Z = tf.squeeze(tf.map_fn(self.conv,(tf.expand_dims(Xaux,1),memW),dtype=tf.float32),axis=1)#tf.nn.convolution(Xaux,memW,self.strides,self.padding)
				#Z = tf.reshape(Z, [self.batch_size, tf.shape(Z)[2],tf.shape(Z)[3],tf.shape(Z)[4]])
				##################################################################################################################################
				#OPTIMIZED LAYER
				strides = [1,self.strides,self.strides,1]
				inp_r, F = reshape(self.X,memW,self.batch_size) #Makes the reshape from [batch,H,W,ch] to [1,H,W,Ch*batch] for input. For filters from [batch,fh,fw,Ch,out_ch]  to
																#[fh,fw,ch*batch,out_ch]
				Z = tf.nn.depthwise_conv2d(
                        inp_r,
                        filter=F,
                        strides=strides,
                        padding=self.padding)
				Z = Z_reshape(Z,memW,self.X,self.padding) #Output shape from convolution is [1,newH,newW,batch*Ch*out_ch] so it is reshaped to [newH,newW,batch,Ch,out_ch]
														  #Where newH and newW are the new image dimensions. This depends on the value of padding
														  #Padding same: newH = H  and newW = W
														  #Padding valid: newH = H-fh+1 and newW = W-fw+1
				Z = tf.transpose(Z, [2, 0, 1, 3, 4]) #Get the property output shape [batch,nH,nW,Ch,out_ch]
				Z = tf.reduce_sum(Z, axis=3)		#Removes the input channel dimension by adding all this elements                                                        
				Z = membias+Z					#Add the bias
			else:
				if(self.isBin=='yes'):
					weights=self.sign(self.W)*self.Werr
				else:
					weights=self.W*self.Werr
				Z = self.bias*self.Berr+tf.nn.convolution(self.X,weights,self.strides,self.padding)				
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
############################AUXILIAR FUNCTIONS##################################################
def reshape(X,F,batch_size): #Used to reshape the input data and the noisy filters
    inp = X
    F = F
    batch_size=batch_size
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]    
    out_channels = tf.shape(F)[-1]
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, channels*batch_size, out_channels]) 
    inp_r = tf.transpose(inp, [1, 2, 0, 3])
    inp_r = tf.reshape(inp_r, [1, H, W, batch_size*channels_img])
    return inp_r, F          
def Z_reshape(Z,F,X,padding): #Used to reshape the output of the layer
    batch_size=tf.shape(X)[0]
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]    
    out_channels = tf.shape(F)[-1]
    if padding == "SAME":
        out = tf.reshape(Z, [H, W, batch_size, channels, out_channels])
    if padding == "VALID":
        out = tf.reshape(Z, [H-fh+1, W-fw+1, batch_size, channels, out_channels])
    return out         
