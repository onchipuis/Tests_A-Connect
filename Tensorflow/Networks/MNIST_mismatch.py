import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import FC_quant
from Layers import DropConnect
from Layers import fullyconnected
from Layers import Conv
import aconnect.layers as layers
#This scripts define the different network architecture for the training and testing.
"""
Model 0: Is a regular model using keras layers. This model was created only for learning 
purposes
Model 1: The same model 0 but this time we have the dropout layer.
Model 2: Created using a custom dropconnect layer that is was not finished (does not work)
Model 3: First model created to test the A-Connect methodology.
Model 4: Custom layer with A-Connect for binary weights
Model 5: Regular conv network with keras layers for comparison purposes.
Model 6: CUstom convolutional network
Model 7: Convolutional neural network with A-Connect
"""
def Test_MNIST(opt,imgsize=[28,28],Wstd=0,Bstd=0,isBin="no",pool=1000):

	#Keras dense network with no regularization
	if(opt==0):

		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),			
			tf.keras.layers.Softmax()
		])
		

		return model

	#Keras dense network with dropout
	
	elif(opt==1):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()
		])
		

		return model

	
	#Custom Dropconnect layer with mismatch
	
	if(opt==2):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),
			DropConnect.DropConnect(128,0.5),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			DropConnect.DropConnect(10),
			tf.keras.layers.Softmax()
		])
		

		return model
	
	#A-Connect with mismatch, no binarization
	if(opt==3):
		
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),	
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=128),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=128),
=======
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=8),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=8),
=======
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=2),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=2),
=======
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=64),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=64),
=======
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=32),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=32),
=======
<<<<<<< HEAD
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=16),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=16),
=======
			layers.FC_AConnect(128,Wstd,Bstd,isBin, pool=1),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin, pool=1),
>>>>>>> 3cbfc790f8f4f0427e4cf66f7257a79b6af8f8ae
>>>>>>> 80edbcca7f94984d2999f052d184d1fcc5464d74
>>>>>>> 7438869915bb6b1a28f44c520075a3a7d3776d47
>>>>>>> 6e8ce98698ed07143340b86d6382efc37479386f
>>>>>>> 8d1d24811413899458341a7116b38fd77a162812
>>>>>>> bfac657704749c693732920857becd32528a7665
			tf.keras.layers.Softmax()
		])

		return model	

	#Custom FC layer with weights binarization.
	if(opt==4):
		
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),	
			FC_quant.FC_quant(128,isBin),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			FC_quant.FC_quant(10,isBin),
			tf.keras.layers.Softmax()
		])

		return model	
	#Convolutional network 
	if(opt==5):
	
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=imgsize),
			tf.keras.layers.Reshape((imgsize[0],imgsize[1],1)),
			tf.keras.layers.Conv2D(8, kernel_size=(5,5), padding ='same'),
			tf.keras.layers.BatchNormalization(name='BN1'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(name='BN2'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()
		])
		
		return model
	#Custom Convolutional Network
	
	if(opt==6):
	
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=imgsize),
			tf.keras.layers.Reshape((imgsize[0],imgsize[1],1)),
			Conv.Conv(8, kernel_size=(5,5), padding ="valid"),
			tf.keras.layers.BatchNormalization(name='BN1'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(name='BN2'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()
		])
		
		return model
   #Convolutional network with Aconnect		
	if(opt==7):
	
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=imgsize),
			tf.keras.layers.Reshape((imgsize[0],imgsize[1],1)),
			layers.Conv_AConnect(8, kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin, padding ="SAME"),
			tf.keras.layers.BatchNormalization(name='BN1'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Flatten(),
			layers.Conv_AConnect(128,Wstd,Bstd,isBin),
			tf.keras.layers.BatchNormalization(name='BN2'),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin),
			tf.keras.layers.Softmax()
		])
		
		
		return model		
	
	
