"""
Script for training and testing AlexNet with A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import aconnect.layers as layers
import aconnect.scripts as scripts
from datetime import datetime
import numpy as np
import tensorflow as tf
import numpy
import time
from keras.callbacks import LearningRateScheduler
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
                        tf.keras.layers.InputLayer(input_shape=[32,32,3]),
                        tf.keras.layers.experimental.preprocessing.Resizing(227,227),           
                        tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',padding="valid"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="valid"),
                        tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="valid"),
                        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="valid"),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(4096, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(4096, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        #tf.keras.layers.Dense(512, activation='relu'),
                        #tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(10, activation='softmax')
            ])
        else:

                model = tf.keras.models.Sequential([
                        tf.keras.layers.InputLayer(input_shape=[32,32,3]),
                        tf.keras.layers.experimental.preprocessing.Resizing(227,227),    
                        layers.Conv_AConnect(filters=96, kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd,pool=8, strides=4,padding="VALID",Op=1,Slice=1,d_type=tf.dtypes.float16),       
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),               
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                        layers.Conv_AConnect(filters=256, kernel_size=(5,5), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),       
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),                   
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                        layers.Conv_AConnect(filters=384, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),        
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),                   
                        layers.Conv_AConnect(filters=384, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),      
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),                      
                        layers.Conv_AConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),        
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),                      
                        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                        tf.keras.layers.Flatten(),
                        layers.FC_AConnect(1024, Wstd=Wstd,Bstd=Bstd,pool=8,d_type=tf.dtypes.float16),   
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),                      
                        tf.keras.layers.Dropout(0.5),
                        layers.FC_AConnect(1024, Wstd=Wstd,Bstd=Bstd,pool=8,d_type=tf.dtypes.float16),      
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),    
                        layers.FC_AConnect(512, Wstd=Wstd,Bstd=Bstd,pool=8,d_type=tf.dtypes.float16),        
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.BatchNormalization(),    
                        layers.FC_AConnect(10, Wstd=Wstd,Bstd=Bstd,pool=8,d_type=tf.dtypes.float16),
                        tf.keras.layers.Softmax()
            ])


        return model
        
