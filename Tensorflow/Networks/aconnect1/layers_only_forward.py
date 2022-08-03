import tensorflow as tf
import numpy as np
import math
############ This layer was made using the template provided by Keras. For more info, go to the official site.
"""
Fully Connected layer with A-Connect
INPUT ARGUMENTS:
-output_size: output_size is the number of neurons of the layer
-Wstd: Wstd standard deviation of the weights(number between 0-1. By default is 0)
-Bstd: Bstd standard deviation of the bias(number between 0-1. By default is 0)
-isBin: if the layer will binarize the weights(String yes or no. By default is no)
-pool: Number of error that you want to use
-d_type: Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
-weights_regularizer: Weights regularizer. Default is None
-bias_regularizer: Bias regularizer. Default is None
"""

class FC_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                output_size,
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                isQuant=["no","no"],
                bw=[1,1],
                pool=0,
                d_type=tf.dtypes.float16,
                weights_regularizer=None,
                bias_regularizer=None,
                **kwargs): #__init__ method is the first method used for an object in python to initialize the ...

                super(FC_AConnect, self).__init__()                                                             #...object attributes
                self.output_size = output_size                                                                  #output_size is the number of neurons of the layer
                self.Wstd = Wstd                                                                                                #Wstd standard deviation of the weights(number between 0-1. By default is 0)
                self.Bstd = Bstd                                                                                                #Bstd standard deviation of the bias(number between 0-1. By default is 0)
                self.errDistr = errDistr                                         #Distribution followed by the error matrices
                self.isQuant = isQuant                                           #if the layer will binarize the weights, bias or both (list [weights_quat (yes or no) , bias_quant (yes or no)]. By default is ["no","no"])
                self.bw = bw                                                     #Number of bits of weights and bias quantization (List [bw_weights, bw_bias]. By default is [1,1])
                self.pool = pool                                                #Number of error that you want to use
                self.d_type = d_type                                            #Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
                self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
                self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
                self.validate_init()
        
        def build(self,input_shape):                                                             #This method is used for initialize the layer variables that depend on input_shape
                                                                                                    #input_shape is automatically computed by tensorflow
                self.W = self.add_weight("W",
                                        shape = [int(input_shape[-1]),self.output_size], #Weights matrix
                                        initializer = "glorot_uniform",
                                        dtype = self.d_type,
                                        regularizer = self.weights_regularizer,
                                        trainable=True)

                self.bias = self.add_weight("bias",
                                        shape = [self.output_size,],                                    #Bias vector
                                        initializer = "zeros",
                                        dtype = self.d_type,
                                        regularizer = self.bias_regularizer,
                                        trainable=True)

                if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
                        if(self.Bstd != 0):
                                self.infBerr = Merr_distr([self.output_size],self.Bstd,self.d_type,self.errDistr)
                                self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
                                                                                                         #This was necessary to change the error matrix/array when Monte Carlo was running.
                        else:
                                self.Berr = tf.constant(1,dtype=self.d_type)
                        if(self.Wstd != 0):
                                self.infWerr = Merr_distr([int(input_shape[-1]),self.output_size],self.Wstd,self.d_type,self.errDistr)
                                self.infWerr = self.infWerr.numpy()
                        else:
                                self.Werr = tf.constant(1,dtype=self.d_type)
                else:
                        self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
                        self.Berr = tf.constant(1,dtype=self.d_type)
                super(FC_AConnect, self).build(input_shape)

        def call(self, X, training=None): #With call we can define all the operations that the layer do in the forward propagation.
                self.X = tf.cast(X, dtype=self.d_type)
                row = tf.shape(self.X)[-1]
                self.batch_size = tf.shape(self.X)[0] #Numpy arrays and tensors have the number of array/tensor in the first dimension.
                                                      #i.e. a tensor with this shape [1000,784,128] are 1000 matrix of [784,128].
                                                      #Then the batch_size of the input data also is the first dimension.
                #Quantize the weights
                if(self.isQuant[0]=="yes"):
                    weights = self.LQuant(self.W)    
                else:
                    weights = self.W
                #Quantize the biases
                if(self.isQuant[1]=="yes"):
                    bias = self.LQuant(self.bias)
                else:
                    bias = self.bias
                #This code will train the network. For inference, please go to the else part
                if(training):
                    if(self.Wstd != 0 or self.Bstd != 0):
                            
                        if(self.Wstd !=0):
                            Werr = Merr_distr([self.pool,tf.cast(row,tf.int32),self.output_size],self.Wstd,self.d_type,self.errDistr)
                        else:
                            Werr = self.Werr

                        if(self.Bstd !=0):  
                            Berr = Merr_distr([self.pool,self.output_size],self.Bstd,self.d_type,self.errDistr)
                        else:
                            Berr = self.Berr

                        newBatch = tf.cast(tf.floor(tf.cast(self.batch_size/self.pool,dtype=tf.float16)),dtype=tf.int32)
                        werr_aux = self.custom_mult(weights,Werr[0])
                        berr_aux = self.custom_mult(bias,Berr[0])
                        Z = tf.matmul(self.X[0:newBatch], werr_aux)  #Matrix multiplication between input and mask. With output shape [batchsize,1,128]
                        Z = tf.reshape(Z,[newBatch,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
                        Z = tf.add(Z,berr_aux) #FInally, we add the bias error mask
                        for i in range(self.pool-1):
                            werr_aux = self.custom_mult(weights,Werr[i+1])
                            berr_aux = self.custom_mult(bias,Berr[i+1])
                            Z1 = tf.matmul(self.X[(i+1)*newBatch:(i+2)*newBatch],werr_aux)
                            Z1 = tf.add(Z1,berr_aux)
                            Z = tf.concat([Z,Z1],axis=0)

                    else:
                        #Custom FC layer operation when we don't have Wstd or Bstd.
                        w = weights*self.Werr
                        b = bias*self.Berr
                        Z = tf.add(tf.matmul(self.X,w),b) 

                else:
                    #This part of the code will be executed during the inference
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
                    #Custom FC layer operation
                    w = weights*Werr
                    b = bias*Berr
                    Z = tf.add(tf.matmul(self.X, w), b)
                        
                Z = self.LQuant(Z)
                return Z
        
        #THis is only for saving purposes. Does not affect the layer performance.
        def get_config(self):
                config = super(FC_AConnect, self).get_config()
                config.update({
                        'output_size': self.output_size,
                        'Wstd': self.Wstd,
                        'Bstd': self.Bstd,
                        'isQuant': self.isQuant,
                        'bw': self.bw,
                        'pool' : self.pool,
                        'd_type': self.d_type,
                        'errDistr ': self.errDistr,
                        'weights_regularizer': self.weights_regularizer,
                        'bias_regularizer' : self.bias_regularizer})
                return config

        def validate_init(self):
                if self.output_size <= 0:
                    raise ValueError('Unable to build a Dense layer with 0 or negative dimension. ' 'Output size: %d' %(self.output_size,))
                if self.Wstd > 1 or self.Wstd < 0:
                    raise ValueError('Wstd must be a number between 0 and 1. \n' 'Found %d' %(self.Wstd,))
                if self.Bstd > 1 or self.Bstd < 0:
                    raise ValueError('Bstd must be a number between 0 and 1. \n' 'Found %d' %(self.Bstd,))
                if not isinstance(self.errDistr, str):
                    raise TypeError('errDistr must be a string. Only two distributions supported: "normal", "lognormal"'
                            'Found %s' %(type(self.errDistr),))
                if not isinstance(self.isQuant, list):
                    raise TypeError('isQuant must be a list, ["yes","yes"] , ["yes","no"], ["no","yes"] or ["no","no"]. ' 'Found %s' %(type(self.isQuant),))
                if self.pool is not None and not isinstance(self.pool, int):
                    raise TypeError('pool must be a integer. ' 'Found %s' %(type(self.pool),))
        
        @tf.custom_gradient
        def LQuant(self,x):      # Gradient function for weights quantization
            y, grad = Quant_custom(x,self)
            return y,grad
        
        @tf.custom_gradient
        def custom_mult(self,x,xerr):      # Gradient function for weights quantization
            y,grad = mult_custom(x,xerr)
            return y,grad
        
###########################################################################################################3
"""
Convolutional layer with A-Connect
INPUT ARGUMENTS:
-filters: Number of filter that you want to use during the convolution.(Also known as output channels)
-kernel_size: List with the dimension of the filter. e.g. [3,3]. It must be less than the input data size
-Wstd and Bstd: Weights and bias standard deviation for training
-pool: Number of error matrices that you want to use.
-isBin: string yes or no, whenever you want binary weights
-strides: Number of strides (or steps) that the filter moves during the convolution
-padding: "SAME" or "VALID". If you want to keep the same size or reduce it.
-Op: 1 or 2. Which way to do the convolution you want to use. The first option is slower but has less memory cosumption and the second one is faster
but consumes a lot of memory.
-d_type: Type of the parameters that the layers will create. Supports fp16, fp32 and fp64
"""
class Conv_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                filters,
                kernel_size,
                strides=1,
                padding="VALID",
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                pool=0,
                isQuant=['no','no'],
                bw=[1,1],
                Op=1,
                d_type=tf.dtypes.float32,
                weights_regularizer=None,
                bias_regularizer=None,
                **kwargs):

                super(Conv_AConnect, self).__init__()
                self.filters = filters
                self.kernel_size = kernel_size
                self.Wstd = Wstd
                self.Bstd = Bstd
                self.errDistr = errDistr
                self.pool = pool
                self.isQuant = isQuant
                self.bw = bw
                self.strides = strides
                self.padding = padding
                self.Op = Op
                self.d_type = d_type
                self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
                self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
                self.validate_init()
        def build(self,input_shape):
                self.shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB

                self.W = self.add_weight('kernel',
                                          shape = self.shape,
                                          initializer = "glorot_uniform",
                                          dtype=self.d_type,
                                          regularizer = self.weights_regularizer,
                                          trainable=True)
                self.bias = self.add_weight('bias',
                                            shape=(self.filters,),
                                            initializer = 'zeros',
                                            dtype=self.d_type,
                                            regularizer = self.bias_regularizer,
                                            trainable=True)
                if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
                        if(self.Bstd != 0):
                                self.infBerr = Merr_distr([self.filters,],self.Bstd,self.d_type,self.errDistr)
                                self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
                                                                                                         #This was necessary to change the error matrix/array when Monte Carlo was running.

                        else:
                                self.Berr = tf.constant(1,dtype=self.d_type)
                        if(self.Wstd !=0):
                                self.infWerr = Merr_distr(self.shape,self.Wstd,self.d_type,self.errDistr)
                                self.infWerr = self.infWerr.numpy()

                        else:
                                self.Werr = tf.constant(1,dtype=self.d_type)
                else:
                        self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
                        self.Berr = tf.constant(1,dtype=self.d_type)
                super(Conv_AConnect, self).build(input_shape)
        def call(self,X,training):
                self.X = tf.cast(X, dtype=self.d_type)
                self.batch_size = tf.shape(self.X)[0]
                
                #Quantize the weights
                if(self.isQuant[0]=="yes"):
                    weights = self.LQuant(self.W)    
                else:
                    weights = self.W
                #Quantize the biases
                if(self.isQuant[1]=="yes"):
                    bias = self.LQuant(self.bias)
                else:
                    bias = self.bias
                
                if(training):
                    if(self.Wstd != 0 or self.Bstd != 0):
                        if(self.Wstd != 0):
                            Werr = Merr_distr(list((self.pool,))+self.shape,self.Wstd,self.d_type,self.errDistr)
                        else:
                            Werr = self.Werr

                        if(self.Bstd != 0):
                            Berr = Merr_distr([self.pool,self.filters],self.Bstd,self.d_type,self.errDistr)
                        else:
                            Berr = self.Berr

                        newBatch = tf.cast(tf.floor(tf.cast(self.batch_size/self.pool,dtype=tf.float16)),dtype=tf.int32)
                        werr_aux = self.custom_mult(weights,Werr[0])
                        berr_aux = self.custom_mult(bias,Berr[0])
                        Z = tf.nn.conv2d(self.X[0:newBatch],werr_aux,strides=[1,self.strides,self.strides,1],padding=self.padding)
                        Z = tf.add(Z,berr_aux) #FInally, we add the bias error mask
                        for i in range(self.pool-1):
                            werr_aux = self.custom_mult(weights,Werr[i+1])
                            berr_aux = self.custom_mult(bias,Berr[i+1])
                            Z1 = tf.nn.conv2d(self.X[(i+1)*newBatch:(i+2)*newBatch],werr_aux,strides=[1,self.strides,self.strides,1],padding=self.padding)
                            Z1 = tf.add(Z1,berr_aux)
                            Z = tf.concat([Z,Z1],axis=0)
                    else:
                        #Custom Conv layer operation
                        w = weights*self.Werr
                        b = bias*self.Berr
                        Z = b+tf.nn.conv2d(self.X,w,self.strides,self.padding)
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
                    
                    #Custom Conv layer operation
                    w = weights*Werr
                    b = bias*Berr
                    Z = b+tf.nn.conv2d(self.X,w,self.strides,self.padding)
                
                Z = self.LQuant(Z)
                return Z
        
        def validate_init(self):
                if not isinstance(self.filters, int):
                    raise TypeError('filters must be an integer. ' 'Found %s' %(type(self.filters),))
                if self.Wstd > 1 or self.Wstd < 0:
                    raise ValueError('Wstd must be a number between 0 and 1. \n' 'Found %d' %(self.Wstd,))
                if self.Bstd > 1 or self.Bstd < 0:
                    raise ValueError('Bstd must be a number between 0 and 1. \n' 'Found %d' %(self.Bstd,))
                if not isinstance(self.errDistr, str):
                    raise TypeError('errDistr must be a string. Only two distributions supported: "normal", "lognormal"'
                            'Found %s' %(type(self.errDistr),))
                if not isinstance(self.isQuant, list):
                    raise TypeError('isQuant must be a list, ["yes","yes"] , ["yes","no"], ["no","yes"] or ["no","no"]. ' 'Found %s' %(type(self.isQuant),))
                if self.pool is not None and not isinstance(self.pool, int):
                    raise TypeError('pool must be a integer. ' 'Found %s' %(type(self.pool),))
        def get_config(self):
                config = super(Conv_AConnect, self).get_config()
                config.update({
                        'filters': self.filters,
                        'kernel_size': self.kernel_size,
                        'Wstd': self.Wstd,
                        'Bstd': self.Bstd,
                        'errDistr': self.errDistr,
                        'pool': self.pool,
                        'isQuant': self.isQuant,
                        'bw': self.bw,
                        'strides': self.strides,
                        'padding': self.padding,
                        'Op': self.Op,
                        'd_type': self.d_type})
                return config
        
        @tf.custom_gradient
        def LQuant(self,x):      # Gradient function for weights quantization
            y, grad = Quant_custom(x,self)
            return y,grad
        
        @tf.custom_gradient
        def custom_mult(self,x,xerr):      # Gradient function for weights quantization
            y,grad = mult_custom(x,xerr)
            return y,grad
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
    inp_r = tf.reshape(inp_r, [1, H, W, batch_size*channels_img])
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
        return tf.reshape(Z, [tf.floor(tf.cast((H)/strides,dtype=tf.float16)), tf.floor(tf.cast((W)/strides,dtype=tf.float16)), batch_size, channels, out_channels])
    if padding == "VALID":
        return tf.reshape(Z, [tf.floor(tf.cast((H-fh)/strides,dtype=tf.float16))+1, tf.floor(tf.cast((W-fw)/strides,dtype=tf.float16))+1, batch_size, channels, out_channels])
    #return out

def Merr_distr(shape,stddev,dtype,errDistr):
    N =  tf.random.normal(shape=shape,
                        stddev=stddev,
                        dtype=dtype)

    if errDistr == "normal":
      Merr = tf.math.abs(1+N)
    elif errDistr == "lognormal":
      #Merr = tf.math.exp(-N)*np.exp(0.5*np.power(stddev,2))
      Merr = tf.math.exp(-N)
    return Merr

def Quant_custom(x,self):
    
    if x.name == "bias":
        bwidth = self.bw[1]
    elif x.name == "W" or x.name == "kernel":
        bwidth = self.bw[0]
    else:
        bwidth = self.bw[0]
    
    if (bwidth==1):
        y = tf.math.sign(x)
    else:
    
        """
        if len(x.get_shape())<2:
            limit = math.sqrt(6/x.get_shape()[0])
        else:
            limit = math.sqrt(6/(x.get_shape()[0]+x.get_shape()[1]))
        """
        #xStd = tf.math.reduce_std(x)
        #xMean = tf.math.reduce_mean(x)
        #limit = 3*xStd
        #limit = tf.cast(limit,tf.dtypes.float32)
        #limit = 1
    
        xi = tf.cast(x,tf.dtypes.float32)
        xMin = tf.math.reduce_min(xi)
        xMax = tf.math.reduce_max(xi)
        xq = tf.quantization.fake_quant_with_min_max_vars(inputs=xi,min=xMin,max=xMax,num_bits=bwidth)
        y = tf.cast(xq,self.d_type)
    
        """
        xFS = xMax-xMin
        Nlevels = 2**bwidth
        xLSB = xFS/Nlevels
        xq = tf.floor(x/xLSB+1)
        xq = tf.clip_by_value(xq,-Nlevels/2+1,Nlevels/2-1)-0.5
        y = xq*xLSB
        """
        """
        limit = math.sqrt(6/((x.get_shape()[0])+(x.get_shape()[1])))
        y = (tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[0]-1))+1),-(2**(self.bw[0]-1)-1), 2**(self.bw[0]-1)) -0.5)*(2/(2**self.bw[0]-1))*limit
        def grad(dy):
                dydx = tf.multiply(dy,tf.divide(y,x+1e-5))
        """
    
    def grad(dy):
        #e = tf.cast(xLSB,self.d_type)*1e-2
        e = 1e-18
        if (bwidth==1):
            dydx = tf.divide(dy,abs(x)+e)
        else:
            #xe = tf.divide(y,x+1e-12)  # Not working
            #dydx = tf.multiply(dy,xe)
            dydx = dy
        return dydx
    
    return y,grad

def mult_custom(x,xerr):      # Gradient function for weights quantization
    y = x*xerr
    
    def grad(dy):
        #dy_dx = dy*xerr
        #dy_dxerr = dy*x
        dy_dx = dy
        dy_dxerr = dy
        return dy_dx, dy_dxerr
    return y,grad
