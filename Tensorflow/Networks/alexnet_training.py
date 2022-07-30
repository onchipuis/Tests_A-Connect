"""
Script for training AlexNet with or without A-Connect
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import math
import tensorflow as tf
import AlexNet as alexnet
from general_training import general_training
from keras.callbacks import LearningRateScheduler
from aconnect1 import layers, scripts
#from aconnect import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

# LOADING DATASET:
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
(X_train,X_test) = normalization(X_train,X_test)

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.7]   # Define the stddev for training
Conv_pool = [8]
FC_pool = [8]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = ["yes"] 
Wbw = [8]
Bbw = [8]
errDistr = ["lognormal"]
#errDistr = ["normal"]
saveModel = True

model_name = 'AlexNet_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
#net_base = folder_models+'Base.h5'
net_base = folder_models+'Wstd_0_Bstd_0.h5'
model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)
transferLearn = True

# TRAINING PARAMETERS
lr_init = 0.01
momentum = 0.9
batch_size = 256
epochs = 50#100
optimizer = tf.optimizers.SGD(learning_rate=0.0, 
                            momentum=momentum) #Define optimizer
def step_decay (epoch): 
    initial_lrate = lr_init 
    drop = 0.5 
    epochs_drop = 30.0 
    lrate = initial_lrate * math.pow (drop,  math.floor ((1 + epoch) / epochs_drop)) 
    return lrate
lrate = LearningRateScheduler(step_decay)
callbacks = [lrate]


# TRAINING THE MODEL:
general_training(model_int=alexnet.model_creation,isAConnect=isAConnect,
                        model_base=model_base,transferLearn=transferLearn,
                        Wstd_err=Wstd_err,
                        WisQuant=WisQuant,BisQuant=BisQuant,
                        Wbw=Bbw,Bbw=Bbw,
                        Conv_pool=Conv_pool,
                        FC_pool=FC_pool,
                        errDistr=errDistr,
                        input_shape=None,depth=None,namev='',
                        optimizer=optimizer,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        saveModel=saveModel,folder_models=folder_models,
                        folder_results=folder_results)

