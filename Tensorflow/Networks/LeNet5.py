

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
        Conv_pool=8,FC_pool=8,errDistr="normal"):
		
	if(not(isAConnect)):
		model = tf.keras.Sequential([
			InputLayer(input_shape=[32,32]),
			Reshape((32,32,1)),
			Conv2D(6,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
                        BatchNormalization(),           
			#Activation('tanh'),           
                        AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Conv2D(16,kernel_size=(5,5),strides=(1,1),padding="valid",activation="tanh"),
                        BatchNormalization(),           
			#Activation('tanh'),			
                        AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Flatten(),
			Dense(120,activation="tanh"),
                        BatchNormalization(),           
			#Activation('tanh'),           
			Dense(84,activation="tanh"),
                        BatchNormalization(),           
			#Activation('tanh'),           
			Dense(10),
			Softmax()							
		])
	else:
		model = tf.keras.Sequential([
			InputLayer(input_shape=[32,32]),
			Reshape((32,32,1)),
			Conv_AConnect(6,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,Op=2,strides=1,padding="VALID",pool=Conv_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),
			Activation('tanh'),           
			AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Conv_AConnect(16,kernel_size=(5,5),Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,Op=2 ,strides=1,padding="VALID",pool=Conv_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),           
			Activation('tanh'),                       
			AveragePooling2D(pool_size=(2,2),strides=(2,2),padding="valid"),
			Flatten(),
			FC_AConnect(120,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,pool=FC_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),           
			Activation('tanh'),                       
			FC_AConnect(84,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,pool=FC_pool,d_type=tf.dtypes.float16),
                        BatchNormalization(),           
			Activation('tanh'),                       
			FC_AConnect(10,Wstd,Bstd,errDistr=errDistr,isQuant=isQuant,bw=bw,pool=FC_pool,d_type=tf.dtypes.float16),
			Softmax()							
		])		
		
	
	return model
