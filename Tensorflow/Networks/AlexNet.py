"""
Script for training and testing AlexNet with A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""

import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
import aconnect.layers as layers
import aconnect.scripts as scripts
from datetime import datetime
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
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
### TRAINING STAGE ###


def step_decay (epoch): 
   initial_lrate = 0.01 
   drop = 0.5 
   epochs_drop = 30.0 
   lrate = initial_lrate * math.pow (drop,  math.floor ((1 + epoch) / epochs_drop)) 
   return lrate

seed = 7
numpy.random.seed(seed) 
#lrate = LearningRateScheduler (step_decay)


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



model=model_creation(isAConnect=True,Wstd=0.0,Bstd=0.0)
#parametros para el entrenamiento

lrate = LearningRateScheduler(step_decay)

callbacks_list = [lrate]

top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy', dtype=None)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.0,momentum=0.9), metrics=['accuracy',top5])
print(model.summary())




model.fit(train_images, train_labels,
          batch_size=256,epochs=100,
          validation_split=0.2,callbacks=callbacks_list
          )
model.evaluate(test_images,test_labels)    

y_predict =model.predict(test_images)
print("top-1 score:", get_top_n_score(test_labels, y_predict, 1))

y_predict =model.predict(test_images)
print("top-5 score:", get_top_n_score(test_labels, y_predict, 5))

#model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy',top5])#Compile the model


model.save("./Models/AlexNet_05.h5",include_optimizer=True) # MODEL SAVING LOGIC

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))
print('Tiempo de procesamiento (secs): ', time.time()-tic)


### TEST STAGE ###
def step_decay (epoch): 
   initial_lrate = 0.01 
   drop = 0.5 
   epochs_drop = 20.0 
   lrate = initial_lrate * math.pow (drop,  math.floor ((1 + epoch) / epochs_drop)) 
   return lrate

seed = 7
numpy.random.seed(seed) 
lrate = LearningRateScheduler(step_decay)
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy', dtype=None)
Sim_err = [0.7]
name = 'AlexNet_aconnect'                      
string = './Models/'+name+'.h5'
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}
acc=np.zeros([500,1])
for j in range(len(Sim_err)):
    Err = Sim_err[j]
    force = "yes"
    if Err == 0:
        N = 1
    else:
        N = 500
            #####
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))
    now = datetime.now()
    starttime = now.time()
    print('\n\n*******************************************************************************************\n\n')
    print('TESTING NETWORK: ', name)
    print('With simulation error: ', Err)
    print('\n\n*******************************************************************************************')
    
    acc, media = scripts.MonteCarlo(string,test_images, test_labels,N,Err,Err,force, 		  0,name,custom_objects,optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9),loss='sparse_categorical_crossentropy',metrics=['accuracy',top5],top5=True)
    np.savetxt('../Results/'+name+'_simerr_'+str(int(100*Err))+'_'+str(int(100*Err))+'.txt',acc,fmt="%.2f")

    now = datetime.now()
    endtime = now.time()
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))

    print('\n\n*******************************************************************************************')
    print('\n Simulation started at: ',starttime)
    print('Simulation finished at: ', endtime)            


