import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import ConvAConnect
from Layers import AConnect
from Scripts import MCsim
import numpy as np


def VggNet(isAConnect=False,Wstd=0,Bstd=0):
	if(not(isAConnect)):
		model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(224,224), 			
#stack1
			tf.keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
#stack2
			tf.keras.layers.Conv2D(128,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(128,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
#stack3
			tf.keras.layers.Conv2D(256,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(256,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(256,kernel_size=[1,1],padding='same',activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
#stack4
			tf.keras.layers.Conv2D(512,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(512,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(512,kernel_size=[1,1],padding='same',activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),    
#stack5
			tf.keras.layers.Conv2D(512,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(512,kernel_size=[3,3],padding='same',activation='relu'),
			tf.keras.layers.Conv2D(512,kernel_size=[1,1],padding='same',activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
#dense   
			tf.keras.layers.Dense(256,activation='relu'),
			tf.keras.layers.Dense(256,activation='relu'),
			tf.keras.layers.Dense(10,activation=None),
            tf.keras.layers.Softmax()
	    ])
	else:

		model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[32,32,3]),
            tf.keras.layers.experimental.preprocessing.Resizing(227,227), 
 #stack1
			ConvAConnect.ConvAConnect(64,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(64,kernel_size=[3,3],padding="SAME",activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="SAME"),
#stack2
			ConvAConnect.ConvAConnect(128,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(128,kernel_size=[3,3],padding="SAME",activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="SAME"),
#stack3
			ConvAConnect.ConvAConnect(256,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(256,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(256,kernel_size=[1,1],padding="SAME",activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="SAME"),
#stack4
			ConvAConnect.ConvAConnect(512,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(512,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(512,kernel_size=[1,1],padding="SAME",activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="SAME"),    
#stack5
			ConvAConnect.ConvAConnect(512,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(512,kernel_size=[3,3],padding="SAME",activation='relu'),
			ConvAConnect.ConvAConnect(512,kernel_size=[1,1],padding="SAME",activation='relu'),
			tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="SAME"),
#dense   
			AConnect.AConnect(256,activation='relu',Wstd=Wstd,Bstd=Bstd),
			AConnect.AConnect(256,activation='relu',Wstd=Wstd,Bstd=Bstd),
			AConnect.AConnect(10,activation=None,Wstd=Wstd,Bstd=Bstd),
            tf.keras.layers.Softmax()
	    ])


	return model
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                                   horizontal_flip=True, vertical_flip=True)
train_datagen.fit(train_images)
train_data = train_datagen.flow(train_images,train_labels, batch_size = 256)

val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen.fit(test_images)
val_data = val_datagen.flow(test_images, test_labels, batch_size = 256)
"""

model=VggNet(isAConnect=False,Wstd=0.5,Bstd=0.5)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.1,momentum=0.9), metrics=['accuracy'])
print(model.summary())
model.fit(train_images,train_labels,
          batch_size=128,epochs=10,
          validation_split=0.3
          )
model.evaluate(test_images,test_labels)    
#model.save("../Models/VggNet.h5",include_optimizer=True)

#acc=np.zeros([1000,1])
#acc,media=MCsim.MCsim("../Models/VggNet.h5",test_images, test_labels,1000,0.3,0.3,"no","VggNet_30",SRAMsz=[10000,20000],SRAMBsz=[4096],optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy'])