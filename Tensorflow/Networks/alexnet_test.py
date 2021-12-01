"""
Script for testing AlexNet with or without A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import AlexNet as alexnet
import time
from datetime import datetime
from aconnect import layers, scripts
#from keras.callbacks import LearningRateScheduler
tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

# LOADING DATASET:
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()	

#### MODEL TESTING WITH MONTE CARLO STAGE ####
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
    
    acc, media = scripts.MonteCarlo(net=string,Xtest=X_test,Ytest=Y_test,M=N,
            Wstd=Err,Bstd=Err,force=force,Derr=0,net_name=name,
            custom_objects=custom_objects,
            optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],top5=False
            )
    
    #np.savetxt('../Results/'+name+'_simerr_'+str(int(100*Err))+'_'+str(int(100*Err))+'.txt',acc,fmt="%.2f")

    now = datetime.now()
    endtime = now.time()
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))

    print('\n\n*******************************************************************************************')
    print('\n Simulation started at: ',starttime)
    print('Simulation finished at: ', endtime) 
