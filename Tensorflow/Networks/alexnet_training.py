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
from aconnect import layers, scripts
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
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

#### RUN TRAINING FOR DIFFERENT LEVEL OF STOCHASTICITY
#Wstd_err = [0.3, 0.5, 0.7, 0.8]
Wstd_err = [0.7]
Conv_pool = [8,1]
FC_pool = [8,1]
isAConnect = False
#errDistr = "lognormal"
errDistr = "normal"
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

for j in range(len(Wstd_err)):
    for i in range(len(FC_pool)):
        Err = Wstd_err[j]
        ### TRAINING STAGE ###
        # CREATING NN:
        model = alexnet.model_creation(isAConnect=isAConnect,
                                        Wstd=Err,Bstd=Err,
                                        Conv_pool=Conv_pool[i],
                                        FC_pool=FC_pool[i],
                                        errDistr=errDistr)

        def step_decay (epoch): 
           initial_lrate = 0.01 
           drop = 0.5 
           epochs_drop = 30.0 
           lrate = initial_lrate * math.pow (drop,  math.floor ((1 + epoch) / epochs_drop)) 
           return lrate

        #parametros para el entrenamiento
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]


        #TRAINING PARAMETERS
        model.compile(loss='sparse_categorical_crossentropy', 
                optimizer=tf.optimizers.SGD(learning_rate=0.0,momentum=0.9), 
                metrics=['accuracy'])

        # TRAINING
        model.fit(X_train, Y_train,
                    batch_size=256,
                    epochs=2,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks_list,
                    shuffle=True)

        model.evaluate(X_test,Y_test)    

        y_predict =model.predict(X_test)
        elapsed_time = time.time() - start_time
        print("top-1 score:", get_top_n_score(Y_test, y_predict, 1))
        print("Elapsed time: {}".format(hms_string(elapsed_time)))
        print('Tiempo de procesamiento (secs): ', time.time()-tic)

        # SAVE MODEL:
        Werr = str(int(100*Err))
        Nm = str(int(FC_pool[i]))
        name = Nm+'Werr_'+'Wstd_'+ Werr +'_Bstd_'+ Werr + "_" +errDistr+'Distr'
        model.save('./Models/AlexNet_CIFAR10/'+name+'.h5',include_optimizer=True)
