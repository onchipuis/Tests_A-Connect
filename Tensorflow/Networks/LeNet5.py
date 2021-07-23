

#############################Script to define the LeNet-5 models. With and without A-Connect################3
import tensorflow as tf
import numpy as np
import sys
from Layers import ConvAConnect
from Layers import AConnect
from Layers import Conv
from Layers import FC_quant
from Layers import dva_fc
from Layers import dva_conv
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')

def LeNet5(Xtrain=None,Xtest=None,isAConnect=False,Wstd=0,Bstd=0,isBin="no"):
	if(Xtrain is not None):
	    Xtrain = np.pad(Xtrain, ((0,0),(2,2),(2,2)), 'constant')
	if(Xtest is not None):        
	    Xtest = np.pad(Xtest, ((0,0),(2,2),(2,2)), 'constant')
	
	#print("Updated training data shape: {}".format(Xtrain[0].shape))
	#print("Updated test data shape: {}".format(Xtest[0].shape))
		
	if(not(isAConnect)):
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32]),
			tf.keras.layers.Reshape((32,32,1)),
			tf.keras.layers.Conv2D(6,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			#tf.keras.layers.Activation('tanh'),            
            tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Conv2D(16,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			#tf.keras.layers.Activation('tanh'),			
            tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120,activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			#tf.keras.layers.Activation('tanh'),            
			tf.keras.layers.Dense(84,activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			#tf.keras.layers.Activation('tanh'),            
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()							
		])
	else:
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32]),
			tf.keras.layers.Reshape((32,32,1)),
			dva_conv.dva_conv(6,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin,strides=1,padding="VALID"),
            tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('tanh'),            
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			dva_conv.dva_conv(16,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin ,strides=1,padding="VALID"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			tf.keras.layers.Flatten(),
			dva_fc.dva_fc(120,Wstd,Bstd,isBin=isBin),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			dva_fc.dva_fc(84,Wstd,Bstd,isBin=isBin),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			dva_fc.dva_fc(10,Wstd,Bstd,isBin=isBin),
			tf.keras.layers.Softmax()							
		])		
		
	
	return model,Xtrain,Xtest
