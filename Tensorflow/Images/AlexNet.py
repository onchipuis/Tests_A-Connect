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
import tensorflow as tf


def model_creation(isAConnect=False,Wstd=0,Bstd=0):
	if(not(isAConnect)):
		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(227,227),           
			tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(4096, activation='relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(4096, activation='relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
	    ])
	else:

		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(227,227),    
		    ConvAConnect.ConvAConnect(filters=96, kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd, strides=4,padding="VALID",Op=2,Slice=4,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(5,5), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=4,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    ConvAConnect.ConvAConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=4,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=128, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=4,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=32, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=4,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    tf.keras.layers.Flatten(),
		    AConnect.AConnect(512, Wstd=Wstd,Bstd=Bstd,pool=128,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.Dropout(0.5),
		    AConnect.AConnect(512, Wstd=Wstd,Bstd=Bstd,pool=128,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.Dropout(0.5),
		    AConnect.AConnect(10, Wstd=Wstd,Bstd=Bstd,pool=128,d_type=tf.dtypes.float16),
            tf.keras.layers.Softmax()
	    ])


	return model
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
#CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                                   horizontal_flip=True, vertical_flip=True)
train_datagen.fit(train_images)
train_data = train_datagen.flow(train_images,train_labels, batch_size = 256)

val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen.fit(test_images)
val_data = val_datagen.flow(test_images, test_labels, batch_size = 256)


model=model_creation(isAConnect=False,Wstd=0.5,Bstd=0.5)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9), metrics=['accuracy'])
print(model.summary())
model.fit(train_data,
          batch_size=256,epochs=2,
          validation_data=val_data
          )
model.evaluate(test_images,test_labels)    
model.save("./Models/AlexNet.h5",include_optimizer=True)


Sim_err = [0, 0.3, 0.5, 0.7]
name = 'AlexNet'                      
string = './Models/'+name+'.h5'
acc=np.zeros([1000,1])
for j in range(len(Sim_err)):
    Err = Sim_err[j]
    force = "yes"
    if Err == 0:
        N = 1
    else:
        N = 1000
            #####
    print('\n\n*******************************************************************************************\n\n')
    print('TESTING NETWORK: ', name)
    print('With simulation error: ', Err)
    print('\n\n*******************************************************************************************')
    acc, media = MCsim.MCsim(string,test_images, test_labels,N,Err,Err,force,0,name,optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    np.savetxt('../Results/'+name+'_simerr_'+str(int(100*Err))+'_'+str(int(100*Err))+'.txt',acc,fmt="%.2f")

            #####
           
           

#acc,media=MCsim.MCsim("../Models/AlexNet.h5",test_images, test_labels,1000,0.3,0.3,"no","AlexNet_30",SRAMsz=[10000,50000],optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

