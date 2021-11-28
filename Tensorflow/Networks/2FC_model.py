import tensorflow as tf
import numpy as np
import aconnect.layers as layers
#This scripts define the different network architecture for the training and testing.

def build_model(imgsize=[28,28],Wstd=0,Bstd=0,isBin="no",pool=None):

	model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=imgsize),	
			layers.FC_AConnect(128,Wstd,Bstd,isBin,pool=pool),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			layers.FC_AConnect(10,Wstd,Bstd,isBin,pool=pool),
			tf.keras.layers.Softmax()
		])

		
		
	return model		
	
	
