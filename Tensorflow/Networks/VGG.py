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

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import ..aconnect.aconnect as aconnect 
from aconnect.layers import Conv_AConnect, FC_AConnect 
from aconnect import scripts 
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, ReLU, Softmax
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


def model_creation(isAConnect=False,Wstd=0,Bstd=0,FC_pool=16,err_Distr="normal"):
	if(not(isAConnect)):
		model = tf.keras.models.Sequential([
			InputLayer(input_shape=(32,32,3)),
			#tf.keras.layers.experimental.preprocessing.Resizing(64,64),           
			tf.keras.layers.UpSampling2D(),           
			Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), activation='relu',padding="same"),
			BatchNormalization(),
			Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same"),
			BatchNormalization(),
			MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
			Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
			Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
			MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
	       	        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
	       	        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
	       	        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
			MaxPool2D(pool_size=(2,2), strides=(2,2),padding="same"),
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),  
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
			MaxPool2D(pool_size=(2,2), strides=(2,2)),
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),  
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
	       	        Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			BatchNormalization(),
			MaxPool2D(pool_size=(2,2), strides=(2,2)),
			Flatten(),
			Dropout(0.5),
			Dense(256, activation='relu'),
			BatchNormalization(),
			Dropout(0.5),
			Dense(10, activation='softmax')
	    ])
	else:

		model = tf.keras.models.Sequential([
			InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(100,100),    
			#tf.keras.layers.UpSampling2D(),           
		        Conv_AConnect(filters=64, kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=64, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
		        Conv_AConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
			Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",Op=2,d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr, pool=4,padding="SAME",Op=2,d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        Conv_AConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=4, padding="SAME",Op=2,d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		        Flatten(),
		        #Dropout(0.1),
		        FC_AConnect(256, Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=16,d_type=tf.dtypes.float16),
		        BatchNormalization(),
                        ReLU(),
		        #Dropout(0.1),
		        FC_AConnect(10, Wstd=Wstd,Bstd=Bstd, errDistr=errDistr,pool=16,d_type=tf.dtypes.float16),
                        Softmax()
	    ])


	return model


