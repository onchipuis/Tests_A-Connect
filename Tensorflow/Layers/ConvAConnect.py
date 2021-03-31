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
-Op: 1 or 2. Which way to do the convolution you want to use. The first option is slower but has less memory cosumption and the second one is faster
but consumes a lot of memory.
-Slice: Optional parameter. Used to divide the batch into 2,4 or 8 minibatches of size batch/N. This works only when you select 2 in Op.
-d_type: Type of the parameters that the layers will create. Supports fp16, fp32 and fp64 
"""
class ConvAConnect(tf.keras.layers.Layer):
	def __init__(self,filters,kernel_size,strides=1,padding="VALID",Wstd=0,Bstd=0,isBin='no',Op=1,Slice=1,d_type=tf.dtypes.float32,**kwargs):
		super(ConvAConnect, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.Wstd = Wstd
		self.Bstd = Bstd
		self.isBin = isBin
		self.strides = strides
		self.padding = padding
		self.Op = Op
		self.Slice = Slice
		self.d_type = d_type
	def build(self,input_shape):
		self.shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB

		self.W = self.add_weight('kernel',
								  shape = self.shape,
								  initializer = "glorot_uniform",
                                  dtype=self.d_type,
								  trainable=True)				  
		self.bias = self.add_weight('bias',
									shape=(self.filters,),
									initializer = 'zeros',
                                    dtype=self.d_type,
									trainable=True)
		if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
			if(self.Bstd != 0):
				self.infBerr = abs(1+tf.random.normal(shape=[self.filters,],stddev=self.Bstd)) #Bias error vector for inference
				self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
													 #This was necessary to change the error matrix/array when Monte Carlo was running.
				#self.Berr = abs(1+tf.random.normal(shape=[self.pool,self.filters],stddev=self.Bstd,dtype=self.d_type)) #"Pool" of bias error vectors
																	
			else:
				self.Berr = tf.constant(1,dtype=self.d_type)
			if(self.Wstd !=0): 
				self.infWerr = abs(1+tf.random.normal(shape=self.shape,stddev=self.Wstd)) #Weight matrix for inference
				self.infWerr = self.infWerr.numpy()										 
				#self.Werr = abs(1+tf.random.normal(shape=list((self.pool,))+shape,stddev=self.Wstd,dtype=self.d_type)) #"Pool" of weights error matrices. Here I need to add an extra dimension. So I concatenate it. But to concatenate, the two elements must be the same type, in this cases, the two elements must be a list
				#self.Werr = tf.squeeze(self.Werr, axis=0) # Remove the extra dimension
				 
			else:
				self.Werr = tf.constant(1,dtype=self.d_type)
		else:
			self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
			self.Berr = tf.constant(1,dtype=self.d_type)
		super(ConvAConnect, self).build(input_shape)
	def call(self,X,training):
		self.X = tf.cast(X, dtype=self.d_type)       
		self.batch_size = tf.shape(self.X)[0]
		if(training):
			if(self.Wstd != 0 or self.Bstd != 0):
				#ID = range(np.size(self.Werr,0))
				#ID = tf.random.shuffle(ID)
				if(self.isBin=='yes'):
				    weights=self.sign(self.W)
				else:
				    weights=self.W				
				#loc_id = tf.slice(ID,[0],[self.batch_size])
																#All this code works exactly as A-Connect fullyconnected layer.
				#################################WORST OPTION TO DO THE CONVOLUTION###############################################################
				if(self.Op == 1):
					if(self.Wstd != 0):
					    #Werr = tf.gather(self.Werr,[loc_id])
					    Werr = abs(1+tf.random.normal(shape=list((self.batch_size,))+self.shape,stddev=self.Wstd,dtype=self.d_type))#tf.squeeze(Werr, axis=0)
					else:
					    Werr = self.Werr
					weights = tf.expand_dims(weights,axis=0)
					memW = tf.multiply(weights,Werr)
					if(self.Bstd != 0):
					    #Berr = tf.gather(self.Berr, [loc_id])
					    Berr =  abs(1+tf.random.normal(shape=[self.batch_size,self.filters],stddev=self.Bstd,dtype=self.d_type))#tf.squeeze(Berr, axis=0)
					else:
					    Berr = self.Berr
					bias = tf.expand_dims(self.bias,axis=0)
					membias = tf.multiply(bias,Berr)
					membias = tf.reshape(membias,[self.batch_size,1,1,tf.shape(membias)[-1]])                    
					#Xaux = tf.transpose(self.X,[0,1,2,3])#tf.reshape(self.X, [self.batch_size,tf.shape(self.X)[1],tf.shape(self.X)[2],tf.shape(self.X)[3]])
					Z = tf.squeeze(tf.map_fn(self.conv,(tf.expand_dims(self.X,1),memW), parallel_iterations=256,fn_output_signature=self.d_type),axis=1)#tf.nn.convolution(Xaux,memW,self.strides,self.padding)
					Z = tf.reshape(Z, [self.batch_size, tf.shape(Z)[1],tf.shape(Z)[2],tf.shape(Z)[3]])
					Z = Z+membias
				##################################################################################################################################
				#OPTIMIZED LAYER
				else:
					strides = [1,self.strides,self.strides,1]
					if(self.Slice == 2): #Slice the batch into 2 minibatches of size batch/2
					    miniBatch = tf.cast(self.batch_size/2,dtype=tf.int32)                                                                  
					    Z1 = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch
					    Z2 = self.slice_batch(weights,miniBatch,1,strides) #Takes a portion from minibatch:2*minibatch
					    Z = tf.concat([Z1,Z2],axis=0)
					elif(self.Slice == 4):
					    miniBatch = tf.cast(self.batch_size/4,dtype=tf.int32) #Slice the batch into 4 minibatches of size batch/4
					    Z = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch
					    for i in range(3):
					        Z1 = self.slice_batch(weights,miniBatch,i+1,strides) #Takes a portion from i*minibatch:(i+1)*minibatch
					        Z = tf.concat([Z,Z1],axis=0)                                                                                                                                                                        
					elif(self.Slice == 8):
					    miniBatch = tf.cast(self.batch_size/8,dtype=tf.int32) #Slice the batch into 8 minibatches of size batch/8
					    Z = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch                  
					    for i in range(7):
					        Z1 = self.slice_batch(weights,miniBatch,i+1,strides) #Takes a portion from i*minibatch:(i+1)*minibatch
					        Z = tf.concat([Z,Z1],axis=0)                                                                                                                                                              
					else:     
					    if(self.Wstd != 0):
					        #Werr = tf.gather(self.Werr,[loc_id])
					        Werr = abs(1+tf.random.normal(shape=list((self.batch_size,))+self.shape,stddev=self.Wstd,dtype=self.d_type))#tf.squeeze(Werr, axis=0)
					    else:
					        Werr = self.Werr
					    weights = tf.expand_dims(weights,axis=0)
					    memW = tf.multiply(weights,Werr)
					    if(self.Bstd != 0):
					        #Berr = tf.gather(self.Berr, [loc_id])
					        Berr =  abs(1+tf.random.normal(shape=[self.batch_size,self.filters],stddev=self.Bstd,dtype=self.d_type))#tf.squeeze(Berr, axis=0)
					    else:
					        Berr = self.Berr
					    bias = tf.expand_dims(self.bias,axis=0)
					    membias = tf.multiply(bias,Berr)
					    membias = tf.reshape(membias,[self.batch_size,1,1,tf.shape(membias)[-1]])                                                                        
					    inp_r, memW = reshape(self.X,memW) #Makes the reshape from [batch,H,W,ch] to [1,H,W,Ch*batch] for input. For filters from [batch,fh,fw,Ch,out_ch]  to
                                                                        #[fh,fw,ch*batch,out_ch]
					    Z = tf.nn.depthwise_conv2d(
                                inp_r,
                                filter=memW,
                                strides=strides,
                                padding=self.padding)
					    Z = Z_reshape(Z,Werr,self.X,self.padding,self.strides) #Output shape from convolution is [1,newH,newW,batch*Ch*out_ch] so it is reshaped to [newH,newW,batch,Ch,out_ch]
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
	def slice_batch(self,weights,miniBatch,N,strides):
		if(self.Wstd != 0):
			#Werr = tf.gather(self.Werr,[loc_id])
			Werr = abs(1+tf.random.normal(shape=list((miniBatch,))+self.shape,stddev=self.Wstd,dtype=self.d_type))#tf.squeeze(Werr, axis=0)
		else:
			Werr = self.Werr

		weights = tf.expand_dims(weights,axis=0)
		memW = tf.multiply(weights,Werr)
		if(self.Bstd != 0):
			#Berr = tf.gather(self.Berr, [loc_id])
			Berr =  abs(1+tf.random.normal(shape=[miniBatch,self.filters],stddev=self.Bstd,dtype=self.d_type))#tf.squeeze(Berr, axis=0)
		else:
			Berr = self.Berr
		bias = tf.expand_dims(self.bias,axis=0)
		membias = tf.multiply(bias,Berr)
		membias = tf.reshape(membias,[miniBatch,1,1,tf.shape(membias)[-1]])     
		inp_r, memW = reshape(self.X[N*miniBatch:(N+1)*miniBatch],memW)
		Z = tf.nn.depthwise_conv2d(
                                    inp_r,
                                    filter=memW,
                                    strides=strides,
                                    padding=self.padding)
		Z = Z_reshape(Z,Werr,self.X[N*miniBatch:(N+1)*miniBatch],self.padding,self.strides)
		Z = tf.transpose(Z, [2, 0, 1, 3, 4])
		Z = tf.reduce_sum(Z, axis=3)
		Z = Z+membias        
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
            'Op': self.Op,
			'Slice': self.Slice,
			'd_type': self.d_type})
		return config
	@tf.custom_gradient
	def sign(self,x):
		y = tf.math.sign(x)
		def grad(dy):
			dydx = tf.divide(dy,abs(x)+1e-5)
			return dydx
		return y, grad		
############################AUXILIAR FUNCTIONS##################################################
def reshape(X,F): #Used to reshape the input data and the noisy filters
    batch_size=tf.shape(X)[0]    
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]    
    out_channels = tf.shape(F)[-1]
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, channels*batch_size, out_channels]) 
    inp_r = tf.transpose(X, [1, 2, 0, 3])
    inp_r = tf.reshape(X, [1, H, W, batch_size*channels_img])
    return inp_r, F          
def Z_reshape(Z,F,X,padding,strides): #Used to reshape the output of the layer
    batch_size=tf.shape(X)[0]
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]    
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]    
    out_channels = tf.shape(F)[-1]
    #tf.print(fh)    
    if padding == "SAME":
        Z = tf.reshape(Z, [tf.floor(tf.cast((H)/strides,dtype=tf.float16)), tf.floor(tf.cast((W)/strides,dtype=tf.float16)), batch_size, channels, out_channels])
        #Z = tf.reduce_sum(Z, axis=3)        
    if padding == "VALID":
        Z = tf.reshape(Z, [tf.floor(tf.cast((H-fh)/strides,dtype=tf.float16))+1, tf.floor(tf.cast((W-fw)/strides,dtype=tf.float16))+1, batch_size, channels, out_channels])
        #Z = tf.reduce_sum(Z, axis=3)                        
    return Z         
