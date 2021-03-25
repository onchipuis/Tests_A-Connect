import tensorflow as tf
import numpy as np
import time
from Networks import LeNet5
from Scripts import load_ds
from datetime import datetime
from Scripts import MCsim
from Layers import AConnect
from Layers import ConvAConnect
from Layers import Conv
from Layers import FC_quant
identifier = [True]				#Which network you want to train/test True for A-Connect false for normal LeNet
Sim_err = [0,0.3,0.5,0.7]	#Define all the simulation errors
Wstd = [0.3,0.5,0.7] 			#Define the stddev for training
Bstd = Wstd
isBin = "yes"					#Do you want binary weights?
(x_train, y_train), (x_test, y_test) = load_ds.load_ds() #Load dataset
_,x_train,x_test=LeNet5.LeNet5(x_train,x_test)	#Load x_train, x_test with augmented dimensions. i.e. 32x32
x_test = np.float32(x_test) #Convert it to float32

####Training part


for i in range(len(identifier)): #Iterate over the networks
    #print(type(x_test))
    isAConnect = identifier[i] #Which network should be selected
    if isAConnect: #is a network with A-Connect?
        for c in range(len(Wstd)): #Iterate over the Wstd and Bstd for training
            wstd = str(int(100*Wstd[c]))
            bstd = str(int(100*Bstd[c]))
            name = 'AConnect_LeNet5'+'_Wstd_'+wstd+'_Bstd_'+bstd 
            if isBin == "yes": 
                name = name+'_BW'                     
            print("*****************************TRAINING NETWORK*********************")
            print("\n\t\t\t", name)
            model,_,_=LeNet5.LeNet5(isAConnect=isAConnect,Wstd=Wstd[c],Bstd=Bstd[c],isBin=isBin)#Get the model
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)#Define optimizer
            model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])#Compile the model
            print(model.summary())#See the summary
            history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)#Train the model
            acc = history.history['accuracy']#Save the accuracy and the validation accuracy
            val_acc = history.history['val_accuracy']
            string = './Models/'+name+'.h5'#Define the folder and the name of the model to be saved
            model.save(string,include_optimizer=True)#Save the model
            np.savetxt('./Models/Training data/'+name+'_acc'+'.txt',acc,fmt="%.2f")#Save in a txt the accuracy and the validation accuracy for further analysis
            np.savetxt('./Models/Training data/'+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")    
        
    else:
        model,_,_=LeNet5.LeNet5(isAConnect=isAConnect,isBin=isBin)	#Same logic is applied here. But is for normal lenet5
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
        name = 'LeNet5'        
        if isBin == "yes": 
            name = name+'_BW'          
        model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
        print("*****************************TRAINING NETWORK*********************")
        print("\n\t\t\t", name)        
        print(model.summary())
        history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']        
        string = './Models/'+name+'.h5'
        model.save(string,include_optimizer=True)
        np.savetxt('./Models/Training data/'+'LeNet5'+'_acc'+'.txt',acc,fmt="%.2f")
        np.savetxt('./Models/Training data/'+'LeNet5'+'_val_acc'+'.txt',val_acc,fmt="%.2f")  
        

#This part is for inference. During the following lines the MCSim will be executed.
        
for k in range(len(identifier)): #Iterate over the networks
    isAConnect = identifier[k] #Select the network
    if isAConnect:
        for m in range(len(Wstd)): #Iterate over the training Wstd and Bstd
            wstd = str(int(100*Wstd[m]))
            bstd = str(int(100*Bstd[m]))
            name = 'AConnect_LeNet5'+'_Wstd_'+wstd+'_Bstd_'+bstd
            if isBin == "yes":
                name = name+'_BW'                          
            string = './Models/'+name+'.h5'
            custom_objects = {'ConvAConnect':ConvAConnect.ConvAConnect,'AConnect':AConnect.AConnect} #Custom objects for model loading purposes
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
                acc_noisy, media = MCsim.MCsim(string,x_test,y_test,N,Err,Err,force,0,name,custom_objects,SRAMsz=[1024,1024],SRAMBsz=[1024]) #Perform the simulation
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
        custom_objects = {'Conv':Conv.Conv,'FC_quant':FC_quant.FC_quant} #Custom objects for model loading purposes
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
            acc_noisy, media = MCsim.MCsim(string,x_test,y_test,N,Err,Err,force,0,name,custom_objects,SRAMsz=[1024,1024],SRAMBsz=[1024])
            #####
            now = datetime.now()
            endtime = now.time()

            print('\n\n*******************************************************************************************')
            print('\n Simulation started at: ',starttime)
            print('Simulation finished at: ', endtime)        



