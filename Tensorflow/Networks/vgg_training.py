"""
Script for training VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import VGG as vgg
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from aconnect import layers, scripts
#from keras.callbacks import LearningRateScheduler
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
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()    
"""
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=256)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, Y_test, batch_size=256)
"""
##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
model_aux=tf.keras.applications.VGG16(weights="imagenet", include_top=False,input_shape=(32,32,3))

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
#Wstd_err = [0.3,0.5,0.7]   # Define the stddev for training
Wstd_err = [0.3]	    # Define the stddev for training
Conv_pool = [16]
FC_pool = [4]
isBin = ["no"]		    # Do you want binary weights?
#errDistr = "lognormal"
errDistr = ["normal"]
model_name = 'VGG16_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
net = folder_models+'16Werr_Wstd_80_Bstd_80.h5'

# TRAINING PARAMETERS
learning_rate = 0.01
momentum = 0.9
batch_size = 256
epochs = 5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=196,
                decay_rate=0.9,
                staircase=True)
optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, 
                            momentum=momentum) #Define optimizer
            
### TRAINING
for d in range(len(isAConnect)): #Iterate over the networks
    if isAConnect[d]: #is a network with A-Connect?
        Wstd_aux = Wstd_err
        FC_pool_aux = FC_pool
        Conv_pool_aux = Conv_pool
    else:
        Wstd_aux = [0]
        FC_pool_aux = [0]
        Conv_pool_aux = [0]
        
    for i in range(len(FC_pool_aux)):
        for j in range(len(Wstd_aux)):
            for k in range(len(errDistr)):
                Err = Wstd_aux[j]
                # CREATING NN:
                #model_aux = tf.keras.models.load_model(net,custom_objects = custom_objects)
                model = vgg.model_creation(isAConnect=isAConnect[d],
                                            Wstd=Err,Bstd=Err,
                                            Conv_pool=Conv_pool_aux[i],
                                            FC_pool=FC_pool_aux[i],
                                            errDistr=errDistr[k],
                                            isQuant=['yes','yes'],
                                            bw=[8,8])
               
                ##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
                if isAConnect[d]:
                    model.layers[1].set_weights(model_aux.layers[1].get_weights())
                    model.layers[4].set_weights(model_aux.layers[2].get_weights())
                    model.layers[8].set_weights(model_aux.layers[4].get_weights())
                    model.layers[11].set_weights(model_aux.layers[5].get_weights())
                    model.layers[15].set_weights(model_aux.layers[7].get_weights())
                    model.layers[18].set_weights(model_aux.layers[8].get_weights())
                    model.layers[21].set_weights(model_aux.layers[9].get_weights())
                    model.layers[25].set_weights(model_aux.layers[11].get_weights())
                    model.layers[28].set_weights(model_aux.layers[12].get_weights())
                    model.layers[31].set_weights(model_aux.layers[13].get_weights())
                    model.layers[35].set_weights(model_aux.layers[15].get_weights())
                    model.layers[38].set_weights(model_aux.layers[16].get_weights())
                    model.layers[41].set_weights(model_aux.layers[17].get_weights())
                """
                else:
                    model.layers[1].set_weights(model_aux.layers[1].get_weights())
                    model.layers[3].set_weights(model_aux.layers[2].get_weights())
                    model.layers[6].set_weights(model_aux.layers[4].get_weights())
                    model.layers[8].set_weights(model_aux.layers[5].get_weights())
                    model.layers[11].set_weights(model_aux.layers[7].get_weights())
                    model.layers[13].set_weights(model_aux.layers[8].get_weights())
                    model.layers[15].set_weights(model_aux.layers[9].get_weights())
                    model.layers[18].set_weights(model_aux.layers[11].get_weights())
                    model.layers[20].set_weights(model_aux.layers[12].get_weights())
                    model.layers[22].set_weights(model_aux.layers[13].get_weights())
                    model.layers[25].set_weights(model_aux.layers[15].get_weights())
                    model.layers[27].set_weights(model_aux.layers[16].get_weights())
                    model.layers[29].set_weights(model_aux.layers[17].get_weights())
"""
                # NAME
                Werr = str(int(100*Err))
                Nm = str(int(FC_pool_aux[i]))
                name = Nm+'Werr_'+'Wstd_'+ Werr +'_Bstd_'+ Werr + "_"+errDistr[k]+'Distr'
                
                print("*************************TRAINING NETWORK*********************")
                print("\n\t\t\t", name)
                    
                #TRAINING PARAMETERS
                model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=optimizer, 
                        metrics=['accuracy'])

                # TRAINING
                history = model.fit(X_train, Y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(X_test, Y_test),
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
                string = folder_models + name + '.h5'
                model.save(string,include_optimizer=True)
                #Save in a txt the accuracy and the validation accuracy for further analysis
                np.savetxt(folder_results+name+'_acc'+'.txt',acc,fmt="%.2f") 
                np.savetxt(folder_results+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")
