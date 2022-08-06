import tensorflow as tf
import numpy as np
import math
from aconnect.layers.scripts import Merr_distr,mult_custom,Quant_custom
############ This layer was made using the template provided by Keras. For more info, go to the official site.
"""
Fully Connected layer with A-Connect
INPUT ARGUMENTS:
-output_size: output_size is the number of neurons of the layer
-Wstd: Wstd standard deviation of the weights(number between 0-1. By default is 0)
-Bstd: Bstd standard deviation of the bias(number between 0-1. By default is 0)
-isBin: if the layer will binarize the weights(String yes or no. By default is no)
-pool: Number of error that you want to use
-bwErrProp: True or False flag to enable/disable backward propagation of error matrices
-d_type: Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
-weights_regularizer: Weights regularizer. Default is None
-bias_regularizer: Bias regularizer. Default is None
"""

#@tf.util.tf_export.tf_export("aconnect1.layers.Conv_AConnect")
class FC_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                output_size,
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                isQuant=["no","no"],
                bw=[1,1],
                pool=0,
                bwErrProp = True,
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
                self.bwErrProp = bwErrProp                                      # Do backward propagation of error matrices or not
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

                self.scale = self.add_weight("scale",
                                        shape = [self.output_size,],                                    #Bias vector
                                        initializer = tf.keras.initializers.Constant(0.1),
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
                        
                Z = self.LQuant(Z)*self.scale
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
            y,grad = mult_custom(x,xerr,self.bwErrProp)
            return y,grad
        
###########################################################################################################3

