import tensorflow as tf
import numpy as np
from Networks import MNIST_mismatch
from Scripts import load_ds
from Scripts import MCsim
from Layers import AConnect
import time
from datetime import datetime

Wstd = [0.3,0.5,0.7]
Bstd = Wstd
Sim_err = [0,0.3,0.5,0.7]
isBin = ["no","yes"]
imgSize = [[28,28],[11,11]]
Q = [8,4]
mul = [1,2,4]
for p in range(len(mul)):
    for i in range(len(imgSize)):
        for j in range(len(Q)):
            for k in range(len(isBin)):
                for l in range(len(Wstd)):
                    wstd = str(int(100*Wstd[l]))
                    bstd = str(int(100*Bstd[l]))
                    if(imgSize[i]==[28,28]):                    
                        name = 'AConnect_'+'28x28_Wstd'+'_'+wstd+'_'+'pool_'+str(mul[p])+'xbatch' 
                    else:
                        name = 'AConnect_'+'11x11_Bstd'+'_'+wstd+'_'+'pool_'+str(mul[p])+'xbatch' 
                    if(isBin[k]=="yes"):
                        name = name+'_BW'
                    (x_train,y_train),(x_test,y_test) = load_ds.load_ds(imgSize=imgSize[i],Quant=Q[j])
                    model = MNIST_mismatch.Test_MNIST(3,Wstd=Wstd[l],Bstd=Bstd[l],isBin=isBin[k],pool=256*mul[p])                                                                    
                    optimizer= tf.keras.optimizers.SGD(lr=0.1,momentum=0.9)
                    print("*****************************TRAINING NETWORK*********************")
                    print("\n\t\t\t", name)
                    model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
                    print(model.summary())#See the summary
                    history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)#Train the model
                    acc = history.history['accuracy']#Save the accuracy and the validation accuracy
                    val_acc = history.history['val_accuracy']
                    string = './Models/'+name+'.h5'#Define the folder and the name of the model to be saved
                    model.save(string,include_optimizer=True)#Save the model
                    np.savetxt('./Models/Training data/'+name+'_acc'+'.txt',acc,fmt="%.2f")#Save in a txt the accuracy and the validation accuracy for further analysis
                    np.savetxt('./Models/Training data/'+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")

for p in range(len(mul)):
    for i in range(len(imgSize)):
        for j in range(len(Q)):
            for k in range(len(isBin)):
                for l in range(len(Wstd)):
                    wstd = str(int(100*Wstd[l]))
                    bstd = str(int(100*Bstd[l]))
                    if(imgSize[i]==[28,28]):                    
                        name = 'AConnect_'+'28x28_Wstd'+'_'+wstd+'_'+'pool_'+str(mul[p])+'xbatch' 
                    else:
                        name = 'AConnect_'+'11x11_Wstd'+'_'+wstd+'_'+'pool_'+str(mul[p])+'xbatch'                                                 
                    if(isBin[k]=="yes"):
                        name = name+'_BW'
                    (x_train,y_train),(x_test,y_test) = load_ds.load_ds(imgSize=imgSize[i],Quant=Q[j])
                    net = './Models/'+name+'.h5'
                    custom_objects = {'AConnect':AConnect.AConnect} #Custom objects for model loading purposes                    
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
                        acc_noisy, media = MCsim.MCsim(net,x_test,y_test,N,Err,Err,force,0,name,custom_objects,SRAMsz=[1024,1024],SRAMBsz=[1024]) #Perform the simulation
                        #For more information about MCSim please go to Scripts/MCsim.py
                        #####
                        now = datetime.now()
                        endtime = now.time()

                        print('\n\n*******************************************************************************************')
                        print('\n Simulation started at: ',starttime)
                        print('Simulation finished at: ', endtime)

