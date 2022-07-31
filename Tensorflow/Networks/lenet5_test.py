import numpy as np
import tensorflow as tf
from general_testing import general_testing
from aconnect1 import layers, scripts
#from aconnect import layers, scripts

# LOADING DATASET:
(X_train, Y_train), (X_test, Y_test) = scripts.load_ds() #Load dataset
X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.float32(X_test) #Convert it to float32

#### MODEL TESTING WITH MONTE CARLO STAGE ####
# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
#Wstd_err = [0.3,0.5,0.7]   # Define the stddev for training
Wstd_err = [0]   # Define the stddev for training
Sim_err = [0,0.3,0.5,0.7]
Conv_pool = [2]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = Wbw
errDistr = ["lognormal"]
#errDistr = ["normal"]
MCsims = 100
acc=np.zeros([500,1])
force = "yes"
force_save = True

model_name = 'LeNet5_MNIST/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name

# TRAINING PARAMETERS
learning_rate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum) #Define optimizer
batch_size = 256
epochs = 30

################################################################
# TESTING THE MODEL:
general_testing(isAConnect=isAConnect,
                Wstd_err=Wstd_err,
                Sim_err=Sim_err,
                WisQuant=WisQuant,BisQuant=BisQuant,
                Wbw=Bbw,Bbw=Bbw,
                Conv_pool=Conv_pool,
                errDistr=errDistr,
                namev='',
                optimizer=optimizer,
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                batch_size=batch_size,
                MCsims=MCsims,force=force,force_save=force_save,
                folder_models=folder_models,
                folder_results=folder_results)

