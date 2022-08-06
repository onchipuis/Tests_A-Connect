import tensorflow as tf
import numpy as np
import math
from aconnect.layers.scripts import Merr_distr,mult_custom,Quant_custom

###########################################################################################################3
"""
Convolutional layer with A-Connect
INPUT ARGUMENTS:
-filters: Number of filter that you want to use during the convolution.(Also known as output channels)
-kernel_size: List with the dimension of the filter. e.g. [3,3]. It must be less than the input data size
-Wstd and Bstd: Weights and bias standard deviation for training
-pool: Number of error matrices that you want to use.
-bwErrProp: True or False flag to enable/disable backward propagation of error matrices
-isBin: string yes or no, whenever you want binary weights
-strides: Number of strides (or steps) that the filter moves during the convolution
-padding: "SAME" or "VALID". If you want to keep the same size or reduce it.
-d_type: Type of the parameters that the layers will create. Supports fp16, fp32 and fp64
"""

#@tf.util.tf_export.tf_export("aconnect1.layers.Conv_AConnect")
class Conv_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                filters,
                kernel_size=(3,3),
                strides=1,
                padding="VALID",
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                pool=0,
                isQuant=['no','no'],
                bw=[1,1],
                bwErrProp = True,
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
                self.bwErrProp = bwErrProp                                      # Do backward propagation of error matrices or not
                self.strides = strides
                self.padding = padding
                self.d_type = d_type
                self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
                self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
                self.validate_init()
        def build(self,input_shape):
                ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Chin] RGB
                self.shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) 

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
                self.scale = self.add_weight("scale",
                                            shape=(self.filters,),
                                            initializer = tf.keras.initializers.Constant(0.1),
                                            dtype = self.d_type,
                                            regularizer = self.bias_regularizer,
                                            trainable=True)
                #If the layer will take into account the standard deviation of the weights or the std of 
                #the bias or both
                if(self.Wstd != 0 or self.Bstd != 0):
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
                        for i in range(self.pool):
                            werr_aux = self.custom_mult(weights,Werr[i])
                            berr_aux = self.custom_mult(bias,Berr[i])
                            Z1 = tf.nn.conv2d(self.X[(i)*newBatch:(i+1)*newBatch],
                                                werr_aux,strides=[1,self.strides,self.strides,1],
                                                padding=self.padding)
                            Z1 = tf.add(Z1,berr_aux)
                            if i==0:
                                Z = Z1
                            else:
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
                
                Z = self.LQuant(Z)*self.scale
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

