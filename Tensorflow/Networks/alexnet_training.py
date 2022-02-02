"""
Script for training AlexNet with or without A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import math
import tensorflow as tf
import AlexNet as alexnet
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from aconnect1 import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

#### TRAINING STAGE #########3
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

# LOADING DATASET:
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
#X_train = X_train/255
#X_test = X_test/255
(X_train,X_test) = normalization(X_train,X_test)

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.3]   # Define the stddev for training
Conv_pool = [2]
FC_pool = Conv_pool
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = ["yes"] 
Wbw = [1]
Bbw = [8]
#errDistr = "lognormal"
errDistr = ["normal"]
saveModel = False
Nlayers = [1,5,9,12,15,20,24,27,30]
Nlayers_base = Nlayers

model_name = 'AlexNet_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
net_base = folder_models+'Base.h5'
model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)

# TRAINING PARAMETERS
lr_init = 0.01
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
        
    for j in range(len(Wstd_aux)):
        if Wstd_aux[j]==0: #is a network with A-Connect?
            FC_pool_aux = [0]
            Conv_pool_aux = [0]
        else:
            FC_pool_aux = FC_pool
            Conv_pool_aux = Conv_pool
        
        for p in range (len(WisQuant)):
            if WisQuant[p]=="yes":
                Wbw_aux = Wbw
                Bbw_aux = Bbw
            else:
                Wbw_aux = [8]
                Bbw_aux = [8]

            for q in range (len(Wbw_aux)):
                for i in range(len(Conv_pool_aux)):
                    for k in range(len(errDistr)):
                        Err = Wstd_aux[j]
                        ### TRAINING STAGE ###
                        # CREATING NN:
                        model = alexnet.model_creation(isAConnect=isAConnect,
                                                        Wstd=Err,Bstd=Err,
                                                        isQuant=[WisQuant[p],BisQuant[p]],
                                                        bw=[Wbw_aux[q],Bbw_aux[q]],
                                                        Conv_pool=Conv_pool_aux[i],
                                                        FC_pool=FC_pool_aux[i],
                                                        errDistr=errDistr[k])
                        ##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
                        """
                        if isAConnect[d]:
                            for m in range(len(Nlayers_base)):
                                model.layers[Nlayers[m]].set_weights(
                                        model_base.layers[Nlayers_base[m]].get_weights()
                                        )
                        """
                        # NAME
                        if isAConnect[d]:
                            Werr = str(int(100*Err))
                            Nm = str(int(Conv_pool_aux[i]))
                            if WisQuant[p] == "yes":
                                bws = str(int(Wbw_aux[q]))
                                quant = bws+'bQuant_'
                            else:
                                quant = ''
                            name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr[k]+'Distr'
                        else:
                            name = 'Base'
                        
                        print("*************************TRAINING NETWORK*********************")
                        print("\n\t\t\t", name)

                        #TRAINING PARAMETERS
                        model.compile(loss='sparse_categorical_crossentropy', 
                                optimizer=optimizer, 
                                metrics=['accuracy'])

                        # TRAINING
                        def step_decay (epoch): 
                            initial_lrate = lr_init 
                            drop = 0.5 
                            epochs_drop = 30.0 
                            lrate = initial_lrate * math.pow (drop,  math.floor ((1 + epoch) / epochs_drop)) 
                            return lrate
                        lrate = LearningRateScheduler(step_decay)
                        callbacks_list = [lrate]
                        
                        history = model.fit(X_train, Y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, Y_test),
                                    callbacks=callbacks_list,
                                    shuffle=True)
                        model.evaluate(X_test,Y_test)    

                        y_predict =model.predict(X_test)
                        elapsed_time = time.time() - start_time
                        print("top-1 score:", get_top_n_score(Y_test, y_predict, 1))
                        print("Elapsed time: {}".format(hms_string(elapsed_time)))
                        print('Tiempo de procesamiento (secs): ', time.time()-tic)
                        #Save the accuracy and the validation accuracy
                        acc = history.history['accuracy'] 
                        val_acc = history.history['val_accuracy']

                        # SAVE MODEL:
                        if saveModel:
                            string = folder_models + name + '.h5'
                            model.save(string,include_optimizer=False)
                            #Save in a txt the accuracy and the validation accuracy for further analysis
                            np.savetxt(folder_results+name+'_acc'+'.txt',acc,fmt="%.2f") 
                            np.savetxt(folder_results+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")
