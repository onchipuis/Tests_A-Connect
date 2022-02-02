"""
Script for training and testing AlexNet with A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training 
stage and then load the model to test it using the Monte Carlo simulation.
"""
import tensorflow as tf
from aconnect1.layers import Conv_AConnect, FC_AConnect 
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, ReLU, Softmax
Xsz = 227

def model_creation(isAConnect=False,Wstd=0,Bstd=0,
        isQuant=["no","no"],bw=[8,8],
        Conv_pool=8,FC_pool=8,errDistr="normal"):
        
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
                        Conv_AConnect(filters=96,kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=4,padding="VALID",Op=1,Slice=1,d_type=tf.dtypes.float16),      
                        BatchNormalization(),              
                        ReLU(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Conv_AConnect(filters=256,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),      
                        BatchNormalization(),                  
                        ReLU(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Conv_AConnect(filters=384,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),       
                        BatchNormalization(),                  
                        ReLU(),
                        Conv_AConnect(filters=384,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),     
                        #Conv_AConnect(filters=384,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),     
                        BatchNormalization(),                     
                        ReLU(),
                        Conv_AConnect(filters=256,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),       
                        #Conv_AConnect(filters=256,kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=Conv_pool,isQuant=isQuant,bw=bw,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),       
                        BatchNormalization(),                     
                        ReLU(),
                        MaxPool2D(pool_size=(3,3),strides=(2,2)),
                        Flatten(),
                        FC_AConnect(1024,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,isQuant=isQuant,bw=bw,d_type=tf.dtypes.float16),  
                        BatchNormalization(),                     
                        ReLU(),
                        Dropout(0.5),
                        FC_AConnect(1024,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,isQuant=isQuant,bw=bw,d_type=tf.dtypes.float16),     
                        BatchNormalization(),   
                        ReLU(),
                        FC_AConnect(512,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,isQuant=isQuant,bw=bw,d_type=tf.dtypes.float16),       
                        BatchNormalization(),   
                        ReLU(),
                        FC_AConnect(10,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,pool=FC_pool,isQuant=isQuant,bw=bw,d_type=tf.dtypes.float16),
                        Softmax()
            ])


        return model
        
