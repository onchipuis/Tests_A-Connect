

#############################Script to define the LeNet-5 models. With and without A-Connect################3
#config = open('config.txt','r')
#folder = config.read()
#sys.path.append(folder)
#sys.path.append(folder+'/Layers/')
import tensorflow as tf
from aconnect1.layers import Conv_AConnect, FC_AConnect 
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, ReLU, Softmax, Reshape, Activation


def model_creation(isAConnect=False,Wstd=0,Bstd=0,
        isQuant=["no","no"],bw=[8,8],
        Conv_pool=2,FC_pool=2,errDistr="normal",
        bwErrProp=True,**kwargs):
		
	if(not(isAConnect)):
		model = tf.keras.Sequential([
			InputLayer(input_shape=[32,32]),
			Reshape((32,32,1)),
			Conv2D(64,kernel_size=(4,4),strides=(1,1),padding="valid",activation="relu"),
                        BatchNormalization(),           
                        MaxPool2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Conv2D(64,kernel_size=(4,4),strides=(1,1),padding="valid",activation="tanh"),
                        BatchNormalization(),           
                        MaxPool2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Flatten(),
			Dense(256,activation="relu"),
                        BatchNormalization(),           
			Dense(64,activation="relu"),
                        BatchNormalization(),           
			Dense(10),
			Softmax()							
		])
	else:
		model = tf.keras.Sequential([
			InputLayer(input_shape=[32,32]),
			Reshape((32,32,1)),
			Conv_AConnect(64,kernel_size=(4,4),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,bwErrProp=bwErrProp,strides=1,padding="VALID",pool=Conv_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),
			Activation('relu'),           
                        MaxPool2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Conv_AConnect(64,kernel_size=(4,4),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,bwErrProp=bwErrProp,strides=1,padding="VALID",pool=Conv_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),
			Activation('relu'),           
                        MaxPool2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Flatten(),
			FC_AConnect(256,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,bwErrProp=bwErrProp,pool=FC_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),           
			Activation('relu'),                       
			FC_AConnect(64,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,bwErrProp=bwErrProp,pool=FC_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),           
			Activation('relu'),                       
			FC_AConnect(10,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,bwErrProp=bwErrProp,pool=FC_pool,d_type=tf.dtypes.float16),
			Softmax()							
		])		
		
	
	return model
