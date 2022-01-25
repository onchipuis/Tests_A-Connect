"""
Script for training and testing AlexNet with A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training 
stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from aconnect.layers import Conv_AConnect, FC_AConnect 
from aconnect import scripts 
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, ReLU, Softmax
from keras.callbacks import LearningRateScheduler
tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"
Xsz = 227

def model_creation(isAConnect=False,Wstd=0,Bstd=0,Conv_pool=8,FC_pool=8,errDistr="normal"):
        if(not(isAConnect)):
                model = tf.keras.models.Sequential([
                        InputLayer(input_shape=[32,32,3]),
                        tf.keras.layers.experimental.preprocessing.Resizing(Xsz,Xsz),           
                        Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),activation='relu',padding="valid"),
                        BatchNormalization(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2),padding="valid"),
                        Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),activation='relu',padding="same"),
                        BatchNormalization(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2),padding="valid"),
                        Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding="same"),
                        BatchNormalization(),
                        Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding="same"),
                        BatchNormalization(),
                        Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation='relu',padding="same"),
                        BatchNormalization(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2),padding="valid"),
                        Flatten(),
                        Dense(4096,activation='relu'),
                        Dropout(0.5),
                        Dense(4096,activation='relu'),
                        Dropout(0.5),
                        #Dense(512,activation='relu'),
                        #Dropout(0.5),
                        Dense(10,activation='softmax')
            ])
        else:

                model = tf.keras.models.Sequential([
                        InputLayer(input_shape=[32,32,3]),
                        tf.keras.layers.experimental.preprocessing.Resizing(Xsz,Xsz),   
                        Conv_AConnect(filters=96,kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,strides=4,padding="VALID",Op=1,Slice=1,d_type=tf.dtypes.float16),      
                        ReLU(),
                        BatchNormalization(),              
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Conv_AConnect(filters=256,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),      
                        ReLU(),
                        BatchNormalization(),                  
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Conv_AConnect(filters=384,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),       
                        ReLU(),
                        BatchNormalization(),                  
                        Conv_AConnect(filters=384,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),     
                        ReLU(),
                        BatchNormalization(),                     
                        Conv_AConnect(filters=256,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),       
                        ReLU(),
                        BatchNormalization(),                     
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Flatten(),
                        FC_AConnect(1024,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,d_type=tf.dtypes.float16),  
                        ReLU(),
                        BatchNormalization(),                     
                        Dropout(0.5),
                        FC_AConnect(1024,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,d_type=tf.dtypes.float16),     
                        ReLU(),
                        BatchNormalization(),   
                        FC_AConnect(512,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,d_type=tf.dtypes.float16),       
                        ReLU(),
                        BatchNormalization(),   
                        FC_AConnect(10,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,d_type=tf.dtypes.float16),
                        Softmax()
            ])


        return model
        
