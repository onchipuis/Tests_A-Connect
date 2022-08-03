"""
Script for training VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import VGG16 as vgg
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from aconnect1 import layers, scripts
#from aconnect import layers, scripts
#from keras.callbacks import LearningRateScheduler
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

#Extra code to improve model accuracy
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

# LOADING DATASET:
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()    
X_train, X_test = normalization(X_train,X_test)    

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.7]   # Define the stddev for training
Conv_pool = [2]
FC_pool = [2]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = Wbw
errDistr = ["lognormal"]
#errDistr = ["normal"]
saveModel = True
model_name = 'VGG16_CIFAR10/'
folder_models = './Models/'+model_name
if isAConnect[0]:
    net_base = folder_models+'Wstd_0_Bstd_0.h5'
    #net_base = folder_models+'8Werr_Wstd_70_Bstd_70_8bQuant_normalDistr.h5'
    #net_base = folder_models+'8Werr_Wstd_50_Bstd_50_8bQuant_lognormalDistr.h5'
    model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)
    transferLearn = True

# Does include error matrices during backward propagation?
bwErrProp = [True]
if not(bwErrProp):
    model_name = model_name+'ForwNoise_only/' 
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'

# TRAINING PARAMETERS
learning_rate = 0.1
momentum = 0.9
batch_size = 256
epochs = 50
lr_decay = 0#1e-4
lr_drop = 20
"""
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=196,
                decay_rate=0.9,
                staircase=True)
optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, 
                            momentum=momentum) #Define optimizer
"""
def lr_scheduler(epoch):
    if epoch < 50:
        lr = 0.01 * (0.5 ** (epoch // lr_drop))
    else:
        lr = 0.01 * (0.5 ** ((epoch-50) // lr_drop))
    return lr

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)    
optimizer = tf.optimizers.SGD(learning_rate=0.0, 
                            momentum=momentum, nesterov= True, decay = lr_decay) #Define optimizer
            
################################################################
# TRAINING THE MODEL:
general_training(model_int=vgg.model_creation,isAConnect=isAConnect,
                        model_base=model_base,transferLearn=transferLearn,
                        Wstd_err=Wstd_err,
                        WisQuant=WisQuant,BisQuant=BisQuant,
                        Wbw=Bbw,Bbw=Bbw,
                        Conv_pool=Conv_pool,
                        FC_pool=FC_pool,
                        errDistr=errDistr,
                        bwErrProp=bwErrProp,
                        input_shape=input_shape,depth=depth,namev=namev,
                        optimizer=optimizer,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        saveModel=saveModel,folder_models=folder_models,
                        folder_results=folder_results)

