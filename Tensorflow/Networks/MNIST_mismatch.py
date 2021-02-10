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

def Test_MNIST(opt):

	#Keras dense network with no regularization
	if(opt==0):

		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),			
			tf.keras.layers.Softmax()
		])
		
		fname_test = 'MNIST_keras_layers'
		fname_train = 'MNIST_keras_layers_test'

		return model, fname_test, fname_train

	#Keras dense network with dropout
	
	elif(opt==1):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()
		])
		

		fname_test = 'MNIST_dropout_keras_layers'
		fname_train = 'MNIST_dropout_keras_layers_test'

		return model, fname_test, fname_train

	
	#Custom Dropconnect layer with mismatch
	
	if(opt==2):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			DropConnect.DropConnect(128,0.5),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			DropConnect.DropConnect(10),
			tf.keras.layers.Softmax()
		])
		

		fname_test = 'MNIST_Dropconnect_layer'
		fname_train = 'MNIST_Dropconnect_layer_test'

		return model, fname_test, fname_train
	
	#A-Connect with mismatch, no binarization
	if(opt==3):
		
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),	
			AConnect.AConnect(128,0.5),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			AConnect.AConnect(10, 0.5),
			tf.keras.layers.Softmax()
		])
		fname_test = 'MNIST_AConnect_layer'
		fname_train = 'MNIST_AConnect_layer_test'
		return model, fname_test, fname_train		
