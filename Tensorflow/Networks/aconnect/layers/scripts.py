import tensorflow as tf
import numpy as np
import math

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
        e = 1e-5
        if (bwidth==1):
            dydx = tf.divide(dy,abs(x)+e)
        else:
            #xe = tf.divide(y,x+1e-12)  # Not working
            #dydx = tf.multiply(dy,xe)
            dydx = dy
        return dydx
    
    return y,grad

def mult_custom(x,xerr,bwErrProp):      # Gradient function for weights quantization
    y = x*xerr
    
    def grad(dy):
        if bwErrProp:
            dy_dx = dy*xerr
            dy_dxerr = dy*x
        else:
            dy_dx = dy
            dy_dxerr = dy
        return dy_dx, dy_dxerr
    return y,grad
