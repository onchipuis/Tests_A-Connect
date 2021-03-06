import tensorflow as tf
import numpy as np
import sys
from Layers import ConvAConnect
from Layers import AConnect
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')

def LeNet5(Xtrain=None,Xtest=None,isAConnect=False,Wstd=0,Bstd=0,isBin="no"):
	Xtrain = np.pad(Xtrain, ((0,0),(2,2),(2,2)), 'constant')
	Xtest = np.pad(Xtest, ((0,0),(2,2),(2,2)), 'constant')
	
	print("Updated training data shape: {}".format(Xtrain[0].shape))
	print("Updated test data shape: {}".format(Xtest[0].shape))
		
	if(not(isAConnect)):
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32]),
			tf.keras.layers.Reshape((32,32,1)),
			tf.keras.layers.Conv2D(6,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Conv2D(16,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120,activation="tanh"),
			tf.keras.layers.Dense(84,activation="tanh"),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()							
		])
	else:
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32]),
			tf.keras.layers.Reshape((32,32,1)),
			ConvAConnect.ConvAConnect(6,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin,strides=1,padding="VALID",activation="tanh"),
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			ConvAConnect.ConvAConnect(16,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin,strides=1,padding="VALID",activation="tanh"),
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Flatten(),
			AConnect.AConnect(120,Wstd,Bstd,isBin,activation="tanh"),
			AConnect.AConnect(84,Wstd,Bstd,isBin,activation="tanh"),
			AConnect.AConnect(10,Wstd,Bstd,isBin),
			tf.keras.layers.Softmax()							
		])		
		
	
	return model,Xtrain,Xtest
