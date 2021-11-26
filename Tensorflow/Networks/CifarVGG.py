import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import Conv
from Layers import FC_quant
from datetime import datetime
from Layers import ConvAConnect
from Layers import AConnect
from Scripts import MCsim
import numpy as np
import tensorflow as tf
import math
import numpy
import time
import pathlib
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
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
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(32, 32, 3)),
      #tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
			#tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(145,145),           
			tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same"),
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
			tf.keras.layers.Dropout(0.1),
			tf.keras.layers.Dense(256, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.1),
			tf.keras.layers.Dense(10, activation='softmax')
	    ])
	else:

		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(145,145),    
		    ConvAConnect.ConvAConnect(filters=64, kernel_size=(3,3),Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=64, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
		    ConvAConnect.ConvAConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=128, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding="SAME"),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=1,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, pool=8,strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=512, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd,pool=8, strides=1,padding="SAME",Op=2,Slice=1,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),padding="SAME"),
		    tf.keras.layers.Flatten(),
		    tf.keras.layers.Dropout(0.1),
		    AConnect.AConnect(256, Wstd=Wstd,Bstd=Bstd,pool=16,d_type=tf.dtypes.float16),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.Dropout(0.1),
		    AConnect.AConnect(10, Wstd=Wstd,Bstd=Bstd,pool=16,d_type=tf.dtypes.float16),
            tf.keras.layers.Softmax()
	    ])


	return model
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



def get_top_n_score(target, prediction, n):
    #ordeno los indices de menor a mayor probabilidad
    pre_sort_index = np.argsort(prediction)
    #ordeno de mayor probabilidad a menor
    pre_sort_index = pre_sort_index[:,::-1]
    #cojo las n-top predicciones
    pre_top_n = pre_sort_index[:,:n]
    #obtengo el conteo de acierto
    precision = [1 if target[i] in pre_top_n[i] else 0 for i in range(target.shape[0])]
    #se retorna la precision
    return np.mean(precision)

model_aux=tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                                       input_shape=(32,32,3))
                                      


model=model_creation(isAConnect=True,Wstd=0.3,Bstd=0.3)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
print(model.layers[0])
"""
#sinAconnect
model.layers[2].set_weights(model_aux.layers[1].get_weights())
model.layers[4].set_weights(model_aux.layers[2].get_weights())
model.layers[7].set_weights(model_aux.layers[4].get_weights())
model.layers[9].set_weights(model_aux.layers[5].get_weights())
model.layers[12].set_weights(model_aux.layers[7].get_weights())
model.layers[14].set_weights(model_aux.layers[8].get_weights())
model.layers[16].set_weights(model_aux.layers[9].get_weights())
model.layers[19].set_weights(model_aux.layers[11].get_weights())
model.layers[21].set_weights(model_aux.layers[12].get_weights())
model.layers[23].set_weights(model_aux.layers[13].get_weights())
model.layers[26].set_weights(model_aux.layers[15].get_weights())
model.layers[28].set_weights(model_aux.layers[16].get_weights())
model.layers[30].set_weights(model_aux.layers[17].get_weights())


"""
##conAconnect
model.layers[1].set_weights(model_aux.layers[1].get_weights())
model.layers[4].set_weights(model_aux.layers[2].get_weights())
model.layers[8].set_weights(model_aux.layers[4].get_weights())
model.layers[11].set_weights(model_aux.layers[5].get_weights())
model.layers[15].set_weights(model_aux.layers[7].get_weights())
model.layers[18].set_weights(model_aux.layers[8].get_weights())
model.layers[21].set_weights(model_aux.layers[9].get_weights())
model.layers[25].set_weights(model_aux.layers[11].get_weights())
model.layers[28].set_weights(model_aux.layers[12].get_weights())
model.layers[31].set_weights(model_aux.layers[13].get_weights())
model.layers[35].set_weights(model_aux.layers[15].get_weights())
model.layers[38].set_weights(model_aux.layers[16].get_weights())
model.layers[41].set_weights(model_aux.layers[17].get_weights())


print("model aux",model_aux.layers[1].get_weights())

print("model ",model.layers[1].get_weights())
print(model.summary())

model.fit(train_images, train_labels,
          batch_size=256,epochs=20,
          validation_data=(test_images,test_labels),
          )
model.evaluate(test_images,test_labels)    

y_predict =model.predict(test_images)
print("top-1 score:", get_top_n_score(test_labels, y_predict, 1))

y_predict =model.predict(test_images)
print("top-5 score:", get_top_n_score(test_labels, y_predict, 5))
model.save("./Models/AlexNet.h5",include_optimizer=True)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))
print('Tiempo de procesamiento (secs): ', time.time()-tic)

model.save("./Models/CifarVGG.h5",include_optimizer=True)

"""
Sim_err = [0, 0.3, 0.5, 0.7]
name = 'CifarVGG_Aconnect03'                      
string = './Models/'+name+'.h5'
custom_objects = {'ConvAConnect':ConvAConnect.ConvAConnect,'AConnect':AConnect.AConnect}
acc=np.zeros([1000,1])
for j in range(len(Sim_err)):
    Err = Sim_err[j]
    force = "yes"
    if Err == 0:
        N = 1
    else:
        N = 1000
            #####
    now = datetime.now()
    starttime = now.time()
    print('\n\n*******************************************************************************************\n\n')
    print('TESTING NETWORK: ', name)
    print('With simulation error: ', Err)
    print('\n\n*******************************************************************************************')
    acc, media = MCsim.MCsim(string,test_images, test_labels,N,Err,Err,force,0,name,custom_objects,optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    np.savetxt('../Results/'+name+'_simerr_'+str(int(100*Err))+'_'+str(int(100*Err))+'.txt',acc,fmt="%.2f")

    now = datetime.now()
    endtime = now.time()

    print('\n\n*******************************************************************************************')
    print('\n Simulation started at: ',starttime)
    print('Simulation finished at: ', endtime)        

            #####
           
"""       

#acc,media=MCsim.MCsim("../Models/AlexNet.h5",test_images, test_labels,1000,0.3,0.3,"no","AlexNet_30",SRAMsz=[10000,50000],optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

