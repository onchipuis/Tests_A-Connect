import tensorflow as tf
import numpy as np
import math
from .scripts import Merr_distr,mult_custom,Quant_custom

###########################################################################################################3
"""
Convolutional layer with A-Connect
INPUT ARGUMENTS:
-filters: Number of filter that you want to use during the convolution.(Also known as output channels)
-kernel_size: List with the dimension of the filter. e.g. [3,3]. It must be less than the input data size
-strides: Number of strides (or steps) that the filter moves during the convolution
-padding: "SAME" or "VALID". If you want to keep the same size or reduce it.
-Wstd and Bstd: Weights and bias standard deviation for training
-pool: Number of error matrices that you want to use.
-bwErrProp: True or False flag to enable/disable backward propagation of error matrices
-isBin: string yes or no, whenever you want binary weights
-d_type: Type of the parameters that the layers will create. Supports fp16, fp32 and fp64
"""

#@tf.util.tf_export.tf_export("aconnect1.layers.Conv_AConnect")
class Conv_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                filters,
                kernel_size=(3,3),
                strides=1,
                padding="VALID",
                data_format='NHWC',
                dilations=None,
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                pool=0,
                isQuant=['no','no'],
                bw=[1,1],
                bwErrProp = True,
                d_type=tf.dtypes.float16,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.keras.initializers.Constant(0.),
                kernel_regularizer=None,
                bias_regularizer=None,
                *args,**kwargs):

                super(Conv_AConnect, self).__init__()
                self.filters = filters
                self.kernel_size = kernel_size
                self.strides = strides
                self.padding = padding
                self.data_format=data_format
                self.dilations=dilations
                self.Wstd = Wstd
                self.Bstd = Bstd
                self.errDistr = errDistr
                self.pool = pool
                self.isQuant = isQuant
                self.bw = bw
                self.bwErrProp = bwErrProp 
                self.d_type = d_type
                self.use_bias = use_bias
                self.kernel_initializer = kernel_initializer 
                self.bias_initializer = bias_initializer
                self.kernel_regularizer = kernel_regularizer 
                self.bias_regularizer = bias_regularizer
                self.args = args
                self.kwargs = kwargs
                self.validate_init()
        def build(self,input_shape):
                ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Chin] RGB
                if type(self.kernel_size) is int:
                    kernel_size = self.kernel_size,
                else:
                    kernel_size = self.kernel_size
                kernel_size = list(kernel_size)
                
                if len(kernel_size) > 1:
                    self.shape = kernel_size + list((int(input_shape[-1]),self.filters))
                else:
                    self.shape = kernel_size + kernel_size + list((int(input_shape[-1]),self.filters))

                self.W = self.add_weight('kernel',
                                          shape = self.shape,
                                          initializer = self.kernel_initializer,
                                          regularizer = self.kernel_regularizer,
                                          dtype=self.d_type,
                                          trainable=True)
                if self.use_bias:
                    self.bias = self.add_weight('bias',
                                                shape=(self.filters,),
                                                initializer = self.bias_initializer,
                                                regularizer = self.bias_regularizer,
                                                dtype=self.d_type,
                                                trainable=True)
                #If the layer will take into account the standard deviation of the weights or the std of 
                #the bias or both
                if(self.Wstd != 0 or self.Bstd != 0):
                    if self.use_bias:
                        if(self.Bstd != 0):
                            self.infBerr = Merr_distr([self.filters,],self.Bstd,self.d_type,self.errDistr)
                            #It is necessary to convert the tensor to a numpy array.Tensors are constants 
                            #and cannot be changed. Necessary to change the error matrix/array when 
                            #Monte Carlo is running.
                            self.infBerr = self.infBerr.numpy()
                        else:
                            self.Berr = tf.constant(1,dtype=self.d_type)
                    
                    if(self.Wstd !=0):
                        self.infWerr = Merr_distr(self.shape,self.Wstd,self.d_type,self.errDistr)
                        self.infWerr = self.infWerr.numpy()

                    else:
                        self.Werr = tf.constant(1,dtype=self.d_type)
                else:
                    self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
                    if self.use_bias:
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
                if self.use_bias:
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

                        if self.use_bias:
                            if(self.Bstd != 0):
                                Berr = Merr_distr([self.pool,self.filters],self.Bstd,self.d_type,self.errDistr)
                            else:
                                Berr = self.Berr

                        newBatch = tf.cast(tf.floor(tf.cast(self.batch_size/self.pool,dtype=tf.float16)),dtype=tf.int32)
                        for i in range(self.pool):
                            werr_aux = self.custom_mult(weights,Werr[i])
                            Z1 = tf.nn.conv2d(self.X[(i)*newBatch:(i+1)*newBatch],
                                                werr_aux,
                                                strides=self.strides,
                                                padding=self.padding,
                                                data_format=self.data_format,
                                                dilations=self.dilations)
                            if self.use_bias:
                                berr_aux = self.custom_mult(bias,Berr[i])
                                Z1 = tf.add(Z1,berr_aux)
                            if i==0:
                                Z = Z1
                            else: 
                                Z = tf.concat([Z,Z1],axis=0)
                    else:
                        #Custom Conv layer operation
                        w = weights*self.Werr
                        Z = tf.nn.conv2d(self.X,w,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format,
                                        dilations=self.dilations)
                        if self.use_bias:
                            b = bias*self.Berr
                            Z=Z+b
                else:
                    if(self.Wstd != 0 or self.Bstd !=0):
                        if(self.Wstd !=0):
                                Werr = self.infWerr
                        else:
                                Werr = self.Werr
                        if self.use_bias:
                            if(self.Bstd != 0):
                                    Berr = self.infBerr
                            else:
                                    Berr = self.Berr
                    else:
                        Werr = self.Werr
                        if self.use_bias:
                            Berr = self.Berr
                    
                    #Custom Conv layer operation
                    w = weights*Werr
                    Z = tf.nn.conv2d(self.X,w,
                                    strides=self.strides,
                                    padding=self.padding,
                                    data_format=self.data_format,
                                    dilations=self.dilations)
                    if self.use_bias:
                        b = bias*Berr
                        Z=Z+b
                
                #Z = self.LQuant(Z)*self.scale
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
                        'strides': self.strides,
                        'padding': self.padding,
                        'data_format': self.data_format,
                        'dilations': self.dilations,
                        'kernel_size': self.kernel_size,
                        'use_bias': self.use_bias,
                        'Wstd': self.Wstd,
                        'Bstd': self.Bstd,
                        'errDistr': self.errDistr,
                        'pool': self.pool,
                        'isQuant': self.isQuant,
                        'bw': self.bw,
                        'd_type': self.d_type})
                return config
        
        @tf.custom_gradient
        def LQuant(self,x):      # Gradient function for weights quantization
            y, grad = Quant_custom(x,self)
            return y,grad
        
        @tf.custom_gradient
        def custom_mult(self,x,xerr):      # Gradient function for weights quantization
            y,grad = mult_custom(x,xerr,self.bwErrProp)
            return y,grad

###########################################################################################################3

