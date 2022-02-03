"""
Script for testing VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import VGG as vgg
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
#Sim_err = [0, 0.3, 0.5, 0.7 0.8]
#Wstd_err = [0.3, 0.5, 0.7]
pool = [4]
Sim_err = [0,0.3,0.5,0.7]
Wstd_err = [0.3]
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}
acc=np.zeros([500,1])
force = "yes"

for j in range(len(Sim_err)):

    for i in range(len(Wstd_err)):
    
        for l in range(len(pool)):
    
            # Model NAME:
            Werr = int(100*Wstd_err[i])
            Nm = str(int(pool[l]))
            name = "32x32_"+Nm+'Werr_'+'Wstd_'+str(Werr)+'_Bstd_'+str(Werr)+"_normalDistr" 
            string = './Models/VGG16_CIFAR10/'+name+'.h5'
            
            Err = Sim_err[j]
            if Err == 0:
                N = 1
            else:
                N = 100
                    #####
            
            elapsed_time = time.time() - start_time
            print("Elapsed time: {}".format(hms_string(elapsed_time)))
            now = datetime.now()
            starttime = now.time()
            print('\n\n***********************************************************************************\n\n')
            print('TESTING NETWORK: ', name)
            print('With simulation error: ', Err)
            print('\n\n***************************************************************************************')
            
            acc, stats = scripts.MonteCarlo(net=string,Xtest=X_test,Ytest=Y_test,M=N,
                    Wstd=Err,Bstd=Err,force=force,Derr=0,net_name=name,
                    custom_objects=custom_objects,
                    optimizer=tf.optimizers.SGD(learning_rate=0.1,momentum=0.9),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],top5=False,dtype='float16',
                    errDistr="normal"
                    )
            name_sim = name+'_simErr_'+str(int(100*Err))                      
            name_stats = name+'_stats_simErr_'+str(int(100*Err))                      
            np.savetxt('../Results/VGG16_CIFAR10/'+name_sim+'.txt',acc,fmt="%.2f")
            np.savetxt('../Results/VGG16_CIFAR10/'+name_stats+'.txt',stats,fmt="%.2f")

            now = datetime.now()
            endtime = now.time()
            elapsed_time = time.time() - start_time
            print("Elapsed time: {}".format(hms_string(elapsed_time)))

            print('\n\n***************************************************************************************')
            print('\n Simulation started at: ',starttime)
            print('Simulation finished at: ', endtime)        

