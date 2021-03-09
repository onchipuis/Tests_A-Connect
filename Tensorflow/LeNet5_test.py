import tensorflow as tf
import numpy as np
import time
from Networks import LeNet5
from Scripts import load_ds
from datetime import datetime
from Scripts import MCsim
from Layers import AConnect
from Layers import ConvAConnect
identifier = [True]
Sim_err = [0, 0.3, 0.5, 0.7]
Wstd = [0.3, 0.5,0.7]
Bstd = Wstd
isBin = "no"
(x_train, y_train), (x_test, y_test) = load_ds.load_ds()
_,x_train,x_test=LeNet5.LeNet5(x_train,x_test)
x_test = np.float32(x_test)

for i in range(len(identifier)):
    #print(type(x_test))
    isAConnect = identifier[i]
    if isAConnect:
        for c in range(len(Wstd)):
            wstd = str(int(100*Wstd[c]))
            bstd = str(int(100*Bstd[c]))
            name = 'AConnect_LeNet5'+'_Wstd_'+wstd+'_Bstd_'+bstd
            if isBin == "yes":
                name = name+'_BW'                     
            print("*****************************TRAINING NETWORK*********************")
            print("\n\t\t\t", name)
            model,_,_=LeNet5.LeNet5(x_train,x_test,isAConnect=isAConnect,Wstd=Wstd[c],Bstd=Bstd[c],isBin=isBin)
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
            model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
            print(model.summary())
            history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            string = './Models/'+'AConnect_LeNet5'+'_Wstd_'+wstd+'_Bstd_'+bstd+'.h5'
            model.save(string,include_optimizer=True)
            np.savetxt('./Models/Training data/'+name+'_acc'+'.txt',acc,fmt="%.2f")
            np.savetxt('./Models/Training data/'+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")    
        
    else:
        model,x_train,x_test=LeNet5.LeNet5(x_train,x_test,isAConnect=isAConnect,isBin="no")
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
        model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
        print(model.summary())
        history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']        
        string = './Models/'+'LeNet5'+'.h5'
        name = 'LeNet5'
        model.save(string,include_optimizer=True)
        np.savetxt('./Models/Training data/'+'LeNet5'+'_acc'+'.txt',acc,fmt="%.2f")
        np.savetxt('./Models/Training data/'+'LeNet5'+'_val_acc'+'.txt',val_acc,fmt="%.2f")    
for k in range(len(identifier)):
    isAConnect = identifier[k]
    if isAConnect:
        for m in range(len(Wstd)):
            wstd = str(int(100*Wstd[m]))
            bstd = str(int(100*Bstd[m]))
            name = 'AConnect_LeNet5'+'_Wstd_'+wstd+'_Bstd_'+bstd
            if isBin == "yes":
                name = name+'_BW'                          
            string = './Models/'+name+'.h5'
            custom_objects = {'ConvAConnect':ConvAConnect.ConvAConnect,'AConnect':AConnect.AConnect}
            for j in range(len(Sim_err)):
                Err = Sim_err[j]
                if Err != Wstd[m]:
                    force = "yes"
                else:
                    force = "no"
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

    else:      
        string = './Models/'+'LeNet5'+'.h5'
        name = 'LeNet5'
        custom_objects = None
        for j in range(len(Sim_err)):
            Err = Sim_err[j]
            if Err != 0.5:
                force = "yes"
            else:
                force = "no"
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



