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
from Layers import DropLayer
from Layers import fullyconnected
from Layers import AConnect

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

	
