import tensorflow as tf
import numpy as np
from Layers import AConnect
from Layers import ConvAConnect
from Networks import MNIST_mismatch
from Scripts import load_ds
from datetime import datetime
from Scripts import MCsim
Sim_err = [0,0.3,0.5,0.7]	#Define all the simulation errors
Wstd = [0.7] 			#Define the stddev for training
Bstd = Wstd
isBin = ["yes"]					#Do you want binary weights?
(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgSize=[11,11], Quant=4) #Load dataset
for p in range(len(isBin)):
    for c in range(len(Wstd)): #Iterate over the Wstd and Bstd for training
        wstd = str(int(100*Wstd[c]))
        bstd = str(int(100*Bstd[c]))
        name = 'AConnect_11x11_4bits'+'_Wstd_'+wstd+'_Bstd_'+bstd 
        if isBin[p] == "yes": 
            name = name+'_BW'                     
        print("*****************************TRAINING NETWORK*********************")
        print("\n\t\t\t", name)
        model = MNIST_mismatch.Test_MNIST(opt=3,imgsize=[11,11],Wstd=Wstd[c],Bstd=Bstd[c],isBin=isBin[p])#Get the model
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

for k in range(len(isBin)): #Iterate over the networks
    for m in range(len(Wstd)): #Iterate over the training Wstd and Bstd
        wstd = str(int(100*Wstd[m]))
        bstd = str(int(100*Bstd[m]))
        name = 'AConnect_11x11_4bits'+'_Wstd_'+wstd+'_Bstd_'+bstd
        if isBin[k] == "yes":
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
            acc_noisy, media = MCsim.MCsim(string,x_test,y_test,N,Err,Err,force,0,name,custom_objects) #Perform the simulation
            #For more information about MCSim please go to Scripts/MCsim.py
            #####
            now = datetime.now()
            endtime = now.time()

            print('\n\n*******************************************************************************************')
            print('\n Simulation started at: ',starttime)
            print('Simulation finished at: ', endtime)        