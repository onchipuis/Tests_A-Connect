"""
Script for training and testing AlexNet with A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
"""import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')"""

from datetime import datetime
import aconnect.layers as layers
import aconnect.scripts as scripts
import numpy as np
import tensorflow as tf
import time
#import math
#import numpy
#import pathlib
#from keras.callbacks import LearningRateScheduler
#import matplotlib.pyplot as plt
tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def model_creation(isAConnect=False,Wstd=0,Bstd=0):
	if(not(isAConnect)):
		model = tf.keras.models.Sequential([
                        #tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(32, 32, 3)),
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			#tf.keras.layers.experimental.preprocessing.Resizing(145,145),           
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), activation='relu',padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
	       	        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
	       	        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
	       	        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),  
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),  
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
	       	        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(256, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
	    ])
	else:

		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(100,100),    
		        layers.Conv_AConnect(filters=64, kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=64, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
		        layers.Conv_AConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
			layers.Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=4,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        layers.Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=4, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        tf.keras.layers.Flatten(),
		        #tf.keras.layers.Dropout(0.1),
		        layers.FC_AConnect(256, Wstd=Wstd,Bstd=Bstd,pool=16,d_type=tf.dtypes.float16),
		        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
		        #tf.keras.layers.Dropout(0.1),
		        layers.FC_AConnect(10, Wstd=Wstd,Bstd=Bstd,pool=16,d_type=tf.dtypes.float16),
                        tf.keras.layers.Softmax()
	    ])


	return model



