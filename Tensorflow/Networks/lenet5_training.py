import numpy as np
import tensorflow as tf
import LeNet5 as lenet5
import time
from general_training import general_training
from aconnect import layers, scripts

# LOADING DATASET:
(X_train, Y_train), (X_test, Y_test) = scripts.load_ds() #Load dataset
X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.float32(X_test) #Convert it to float32

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.5]   # Define the stddev for training
#Wstd_err = [0]	    # Define the stddev for training
Conv_pool = [2]
FC_pool = Conv_pool
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = [8]
saveModel = True
errDistr = ["lognormal"]
#errDistr = ["normal"]
model_name = 'LeNet5_MNIST/'

# Does include error matrices during backward propagation?
bwErrProp = [True]
if not(bwErrProp[0]):
    model_name = model_name+'ForwNoise_only/' 
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'

# TRAINING PARAMETERS
learning_rate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum) #Define optimizer
batch_size = 256
epochs = 50

# TRAINING THE MODEL:
general_training(model_int=lenet5.model_creation,isAConnect=isAConnect,
                        model_base=None,transferLearn=False,
                        Wstd_err=Wstd_err,
                        WisQuant=WisQuant,BisQuant=BisQuant,
                        Wbw=Bbw,Bbw=Bbw,
                        Conv_pool=Conv_pool,
                        FC_pool=FC_pool,
                        errDistr=errDistr,
                        bwErrProp=bwErrProp,
                        input_shape=None,depth=None,namev='',
                        optimizer=optimizer,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=None,
                        saveModel=saveModel,folder_models=folder_models,
                        folder_results=folder_results)

