import tensorflow as tf
import numpy as np
import time
import LeNet5
from datetime import datetime
import aconnect.layers as layers
import aconnect.scripts as scripts
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

identifier = [False,True]				#Which network you want to train/test True for A-Connect false for normal LeNet
Sim_err = [0,0.3,0.5,0.7]	#Define all the simulation errors
Wstd = [0.3,0.5,0.7]			#Define the stddev for training
Bstd = Wstd
isBin = ["no"]					#Do you want binary weights?
(x_train, y_train), (x_test, y_test) = scripts.load_ds() #Load dataset
_,x_train,x_test=LeNet5.LeNet5(x_train,x_test)	#Load x_train, x_test with augmented dimensions. i.e. 32x32
x_test = np.float32(x_test) #Convert it to float32
epochs = 2
learning_rate = 0.01
momentum = 0.9
batch_size = 256
N = 1 #Number of error matrices to test, only 2^(n-1) size

#This part is for inference. During the following lines the MCSim will be executed.
for d in range(3,N): #Iterate over all the error matrices

	M = 2**(d)
	nMatriz = str(M)
	for k in range(len(identifier)): #Iterate over the networks
	    isAConnect = identifier[k] #Select the network
	    if isAConnect:
	        for m in range(len(Wstd)): #Iterate over the training Wstd and Bstd
	            wstd = str(int(100*Wstd[m]))
	            bstd = str(int(100*Bstd[m]))
	            name = 'LeNet5_'+nMatriz+'Werr'+'_Wstd_'+wstd+'_Bstd_'+bstd
	            if isBin == "yes":
	                name = name+'_BW'
	            string = './Models/LeNet5_MNIST/'+name+'.h5'
	            custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect} #Custom objects for model loading purposes
	            for j in range(len(Sim_err)): #Iterate over the sim error vector
	                Err = Sim_err[j]
	                if Err != Wstd[m]: #If the sim error is different from the training error, do not force the error
	                    force = "yes"
	                else:
	                    force = "no"
	                if Err == 0: #IF the sim error is 0, takes only 1 sample
	                    N = 1
	                else:
	                    N = 1000
	                now = datetime.now()
	                starttime = now.time()
	                #####
	                print('\n\n*******************************************************************************************\n\n')
	                print('TESTING NETWORK: ', name)
	                print('With simulation error: ', Err)
	                print('\n\n*******************************************************************************************')
	                acc_noisy, media = scripts.MonteCarlo(string,x_test,y_test,N,Err,Err,force,0,'../Results/LeNet5_MNIST/Matrices_Test/'+name,custom_objects,
	                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum),loss=['sparse_categorical_crossentropy'],metrics=['accuracy']
                    ,run_model_eagerly=True,evaluate_batch_size=10000) #Perform the simulation
	                #For more information about MCSim please go to Scripts/MCsim.py
	                #####
	                now = datetime.now()
	                endtime = now.time()

	                print('\n\n*******************************************************************************************')
	                print('\n Simulation started at: ',starttime)
	                print('Simulation finished at: ', endtime)

	    else:
	        name = 'LeNet5'
	        if isBin == "yes":
	            name = name+'_BW'
	        string = './Models/'+name+'.h5'
	#        custom_objects = {'Conv':Conv.Conv,'FC_quant':FC_quant.FC_quant} #Custom objects for model loading purposes
	        for j in range(len(Sim_err)):
	            Err = Sim_err[j]
	            force = "yes"
	            if Err == 0:
	                N = 1
	            else:
	                N = 1000
	            now = datetime.now()
	            starttime = now.time()
	            #####
	            print('\n\n*******************************************************************************************\n\n')
	            print('TESTING NETWORK: ', name)
	            print('With simulation error: ', Err)
	            print('\n\n*******************************************************************************************')
	            acc_noisy, media = scripts.MonteCarlo(string,x_test,y_test,N,Err,Err,force,0,'../Results/LeNet5_MNIST/'+name,
	            optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
	            #####
	            now = datetime.now()
	            endtime = now.time()

	            print('\n\n*******************************************************************************************')
	            print('\n Simulation started at: ',starttime)
	            print('Simulation finished at: ', endtime)
