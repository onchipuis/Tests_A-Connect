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
from aconnect1 import layers, scripts
#from keras.callbacks import LearningRateScheduler
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

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
# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.3,0.5,0.7]   # Define the stddev for training
Sim_err = Wstd_err
Conv_pool = [1,2,4,8,16]
isBin = ["no"]		    # Do you want binary weights?
#errDistr = "lognormal"
errDistr = ["normal"]
acc=np.zeros([500,1])
force = "yes"

model_name = 'AlexNet_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name

# TRAINING PARAMETERS
momentum = 0.9
batch_size = 256
epochs = 30
optimizer = tf.optimizers.SGD(learning_rate=0.0, 
                            momentum=momentum) #Define optimizer
for d in range(len(isAConnect)): #Iterate over the networks
    if isAConnect[d]: #is a network with A-Connect?
        Wstd_aux = Wstd_err
        FC_pool_aux = FC_pool
        Conv_pool_aux = Conv_pool
    else:
        Wstd_aux = [0]
        FC_pool_aux = [0]
        Conv_pool_aux = [0]
        
    for i in range(len(Conv_pool_aux)):
        for j in range(len(Wstd_aux)):
            for k in range(len(errDistr)):

                Werr = Wstd_aux[j]
                Err = Werr
                #Err = Sim_err[j]
                # NAME
                if isAConnect[d]:
                    Werr = str(int(100*Werr))
                    Nm = str(int(Conv_pool_aux[i]))
                    name = Nm+'Werr_'+'Wstd_'+ Werr +'_Bstd_'+ Werr + "_"+errDistr[k]+'Distr'
                else:
                    name = 'Base'
                string = folder_models + name + '.h5'
            
                if Err == 0:
                    N = 1
                else:
                    N = 100
                        #####
                
                elapsed_time = time.time() - start_time
                print("Elapsed time: {}".format(hms_string(elapsed_time)))
                now = datetime.now()
                starttime = now.time()
                print('\n\n********************************************************************************\n\n')
                print('TESTING NETWORK: ', name)
                print('With simulation error: ', Err)
                print('\n\n************************************************************************************')
                
                acc, stats = scripts.MonteCarlo(net=string,Xtest=X_test,Ytest=Y_test,M=N,
                        Wstd=Err,Bstd=Err,force=force,Derr=0,net_name=name,
                        custom_objects=custom_objects,
                        optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'],top5=False,dtype='float16',
                        errDistr=errDistr[k]
                        )
                name_sim = name+'_simErr_'+str(int(100*Err))                      
                name_stats = name+'_stats_simErr_'+str(int(100*Err))                      
                np.savetxt(folder_results+name_sim+'.txt',acc,fmt="%.2f")
                np.savetxt(folder_results+name_stats+'.txt',stats,fmt="%.2f")

                now = datetime.now()
                endtime = now.time()
                elapsed_time = time.time() - start_time
                print("Elapsed time: {}".format(hms_string(elapsed_time)))

                print('\n\n***********************************************************************************')
                print('\n Simulation started at: ',starttime)
                print('Simulation finished at: ', endtime)        

