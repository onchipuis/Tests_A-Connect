### Modified by: Ricardo Vergel - Dec 03/2020
###
### A-Connect fully connected layer

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Scripts')
import Scripts 
from Scripts import addMismatch

class AConnect(tf.keras.layers.Layer):
	def __init__(self,output_size,Wstd=0,isBin = "no",**kwargs):
		super(AConnect, self).__init__()
		self.output_size = output_size
		self.Wstd = Wstd
		self.isBin = isBin
		
	def build(self,input_shape):	
		self.W = self.add_weight("W",
										shape = [int(input_shape[-1]),self.output_size],
										initializer = "glorot_uniform",
										trainable=True)

		self.bias = self.add_weight("bias",
										shape = [self.output_size,],
										initializer = "zeros",
										trainable=True)					
		if(self.Wstd != 0):
			self.Berr = tf.random.normal(shape=[1000,self.output_size],stddev=self.Wstd)
			self.Berr = abs(self.Berr.numpy())
			self.Werr = tf.random.normal(shape=[1000,int(input_shape[-1]),self.output_size],stddev=self.Wstd)
			self.Werr = abs(self.Werr.numpy())
		else:
			self.Werr = tf.constant(1,dtype=tf.float32)
			self.Berr = tf.constant(1,dtype=tf.float32)
		super(AConnect, self).build(input_shape)
		
	def call(self, X, training=None):
		self.X = X
		self.batch_size = tf.shape(self.X)[0]
		
		if(training):	
			if(self.Wstd != 0):

				#[self.memW, self.membias] = addMismatch.addMismatch(self)
				ID = range(np.size(self.Werr,0))
				ID = tf.random.shuffle(ID)
				loc_id = tf.slice(ID, [0], [self.batch_size])
				Werr = tf.gather(self.Werr,[loc_id])
				Werr = tf.squeeze(Werr, axis=0)
				self.memW = tf.multiply(self.W,Werr)
				Berr = tf.gather(self.Berr, [loc_id])
				Berr = tf.squeeze(Berr,axis=0)
				self.membias = tf.multiply(self.bias,Berr)
				
				Xaux = tf.reshape(self.X, [self.batch_size,1,tf.shape(self.X)[-1]])
				Z = tf.matmul(Xaux, self.memW)
				Z = tf.reshape(Z,[self.batch_size,tf.shape(Z)[-1]])
				Z = tf.add(Z,self.membias)
			
		
			else:
				Z = tf.add(tf.matmul(self.X,self.W),self.bias)

		else:
		
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:]
				
			else:
				Werr = self.Werr
				Berr = self.Berr
				
			weights = self.W*Werr#tf.math.multiply(self.W,Werr)
			bias = self.bias*Berr#tf.math.multiply(self.bias,Berr)		
			Z = tf.add(tf.matmul(self.X, weights), bias)
					
		return Z
		
	def get_config(self):
		config = super(AConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'isBin': self.isBin})
		return config

