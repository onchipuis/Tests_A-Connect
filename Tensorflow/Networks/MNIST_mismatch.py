import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
import mylib as my
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import FC_quant
from Layers import DropConnect
from Layers import fullyconnected
from Layers import AConnect
from Layers import Conv
from Layers import ConvAConnect

#This scripts define the different network architecture for the training and testing.

def Test_MNIST(opt,imgsize=[28,28],Wstd=0,Bstd=0,isBin="no"):

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
			AConnect.AConnect(128,Wstd,Bstd,isBin),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			AConnect.AConnect(10,Wstd,Bstd,isBin),
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
			ConvAConnect.ConvAConnect(8, kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin, padding ="valid"),
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
		
		return model		
	
	
