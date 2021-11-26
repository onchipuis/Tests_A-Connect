#############################Script to define the LeNet-5 models. With and without A-Connect################3
import tensorflow as tf
import numpy as np
import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
from Layers import ConvAConnect
from Layers import AConnect
from Layers import Conv
from Layers import FC_quant


def LeNet5(Xtrain=None,Xtest=None,isAConnect=False,Wstd=0,Bstd=0,isBin="no"):
	if(Xtrain is not None):
	    Xtrain = np.pad(Xtrain, ((0,0),(2,2),(2,2)), 'constant')
	if(Xtest is not None):        
	    Xtest = np.pad(Xtest, ((0,0),(2,2),(2,2)), 'constant')
	
	#print("Updated training data shape: {}".format(Xtrain[0].shape))
	#print("Updated test data shape: {}".format(Xtest[0].shape))
		
	if(not(isAConnect)):
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.Reshape((32,32,3)),
			#Conv.Conv(6,kernel_size=(5,5),strides=(4,4),isBin=isBin),
            tf.keras.layers.Conv2D(6,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('relu'),            
            tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
			#Conv.Conv(16,kernel_size=(5,5),strides=1,padding="same",isBin=isBin),
            tf.keras.layers.Conv2D(16,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('relu'),			
            tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="same"),
			tf.keras.layers.Flatten(),
			#FC_quant.FC_quant(120,isBin=isBin),
            tf.keras.layers.Dense(120,activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('relu'),            
			#FC_quant.FC_quant(84,isBin=isBin),
            tf.keras.layers.Dense(84,activation="tanh"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('relu'),            
			#FC_quant.FC_quant(10,isBin=isBin),
            tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()							
		])
	else:
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32]),
			tf.keras.layers.Reshape((32,32,1)),
			ConvAConnect.ConvAConnect(6,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin,strides=1,padding="VALID"),
            tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('tanh'),            
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="VALID"),
			ConvAConnect.ConvAConnect(16,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,isBin=isBin ,strides=1,padding="VALID"),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="VALID"),
			tf.keras.layers.Flatten(),
			AConnect.AConnect(120,Wstd,Bstd,isBin=isBin),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			AConnect.AConnect(84,Wstd,Bstd,isBin=isBin),
            tf.keras.layers.BatchNormalization(),            
			tf.keras.layers.Activation('tanh'),                        
			AConnect.AConnect(10,Wstd,Bstd,isBin=isBin),
			tf.keras.layers.Softmax()							
		])		
		
	
	return model
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


model=LeNet5(isAConnect=False,Wstd=0.5,Bstd=0.5)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9), metrics=['accuracy'])
print(model.summary())
model.fit(train_images,train_labels,
          batch_size=128,epochs=50,
          validation_split=0.2
          )
model.evaluate(test_images,test_labels) 