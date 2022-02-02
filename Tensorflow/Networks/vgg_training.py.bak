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
from tensorflow.keras.utils import to_categorical
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
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)   

# prepare data augmentation configuration
""""
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)
"""
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images
    
X_train, X_test = normalization(X_train,X_test)    
sL = 3
Nlayers = [1,3,6,8,11,13,15,18,20,22,25,27,29] #Baseline layer numbers
NlayersBase = [1,2,4,5,7,8,9,11,12,13,15,16,17]

for j in range(len(Nlayers)):
    Nlayers[j] = Nlayers[j] + sL #Shift the layer index due to the preprocessing layers
print(Nlayers)
##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
model_aux=tf.keras.applications.VGG16(weights="imagenet", include_top=False,input_shape=(32,32,3))

# INPUT PARAMTERS:
isAConnect = [False]   # Which network you want to train/test True for A-Connect false for normal LeNet
#Wstd_err = [0.3,0.5,0.7]   # Define the stddev for training
Wstd_err = [0.7]	    # Define the stddev for training
Conv_pool = [16]
FC_pool = [4]
isBin = ["no"]		    # Do you want binary weights?
#errDistr = "lognormal"
errDistr = ["normal"]
model_name = 'VGG16_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
net = folder_models+ 'baseline'#'32x32_16Werr_Wstd_80_Bstd_80.h5'

# TRAINING PARAMETERS
learning_rate = 0.1
momentum = 0.9
batch_size = 256
epochs = 30
lr_decay = 1e-6
lr_drop = 20
"""
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=196,
                decay_rate=0.9,
                staircase=True)"""
                
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))    

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)    
optimizer = tf.optimizers.SGD(learning_rate=learning_rate, 
                            momentum=momentum, decay = lr_decay, nesterov= True) #Define optimizer
            
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
                print(model.summary())
               
                ##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
                
                if isAConnect[d]:
                    for i in range(len(Nlayers)):
                        model.layers[Nlayers[i]].set_weights(model_aux.layers[NlayersBase[i]].get_weights())

                else:
                    for i in range(len(Nlayers)):
                        model.layers[Nlayers[i]].set_weights(model_aux.layers[NlayersBase[i]].get_weights())

                # NAME
                Werr = str(int(100*Err))
                Nm = str(int(FC_pool_aux[i]))
                name = 'baseline_32x32' #+Nm+'Werr_'+'Wstd_'+ Werr +'_Bstd_'+ Werr + "_"+errDistr[k]+'Distr'
                
                print("*************************TRAINING NETWORK*********************")
                print("\n\t\t\t", name)
                    
                #TRAINING PARAMETERS
                model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer, 
                        metrics=['accuracy'])

               
                # TRAINING
                
                history = model.fit(X_train, Y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(X_test, Y_test),
                          callbacks=[reduce_lr],
                          shuffle=True)
                model.evaluate(X_test, Y_test)    

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
                model.save(string,include_optimizer=False)
                #Save in a txt the accuracy and the validation accuracy for further analysis
                np.savetxt(folder_results+name+'_acc'+'.txt',acc,fmt="%.2f") 
                np.savetxt(folder_results+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")
