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
										shape = [int(input_shape[1]),self.output_size],
										initializer = "glorot_uniform",
										trainable=True)

		self.bias = self.add_weight("bias",
										shape = [1,self.output_size],
										initializer = "zeros",
										trainable=True)
									
		if(self.Wstd != 0):
			self.Berr = abs(tf.random.normal(shape=[1000,1,self.output_size],stddev=self.Wstd))
			self.Berr = self.Berr.numpy()
			self.Werr = abs(tf.random.normal(shape=[1000,int(input_shape[1]),self.output_size],stddev=self.Wstd))
			self.Werr = self.Werr.numpy()
		else:
			self.Werr = tf.constant(1,dtype=tf.float32)
			self.Berr = tf.constant(1,dtype=tf.float32)
		super(AConnect, self).build(input_shape)
		
	def call(self, X, training):
		self.X = X
		self.batch_size = tf.shape(self.X)[0]
		
		if(training):	
			if(self.Wstd != 0):
				[self.memW, self.membias] = addMismatch.addMismatch(self)
				Xaux = tf.reshape(self.X, [self.batch_size,1,tf.shape(self.X)[-1]])
				#ID = range(1000)
				#ID =tf.random.shuffle(ID)
				#self.memW = self.W*tf.gather(self.Werr,[ID[0]])
				#self.membias = self.bias*tf.gather(self.Berr,[ID[0]])
				Z = tf.add(tf.matmul(Xaux, self.memW), self.membias)
				Z = tf.reshape(Z,[self.batch_size,tf.shape(Z)[-1]])
		
			else:
				Z = tf.add(tf.matmul(self.X,self.W),self.bias)

		else:
		
			if(self.Wstd != 0):
				Werr = self.Werr[1,:,:]
				Berr = self.Berr[1,:,:]
				
			else:
				Werr = self.Werr
				Berr = self.Berr
				
			weights = tf.math.multiply(self.W,Werr)
			bias = tf.math.multiply(self.bias,Berr)
			Z = tf.add(tf.matmul(self.X, weights), bias)
					
		return Z
		
	def get_config(self):
		config = super(AConnect, self).get_config()
		config.update({
			'output_size': self.output_size,
			'Wstd': self.Wstd,
			'isBin': self.isBin})
		return config

