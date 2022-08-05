"""
Script for testing VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
from general_testing import general_testing

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

#### MODEL TESTING WITH MONTE CARLO STAGE ####
# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.7]   # Define the stddev for training
Sim_err = [0.7]
Conv_pool = [1]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = Wbw
#errDistr = ["lognormal"]
errDistr = ["normal"]
MCsims = 100
acc=np.zeros([500,1])
force = "yes"
force_save = True
namev=''

model_name = 'ResNet20_CIFAR10/'
# Does include error matrices during backward propagation?
bwErrProp = [True]
if not(bwErrProp[0]):
    model_name = model_name+'ForwNoise_only/' 
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name

# TRAINING PARAMETERS
learning_rate = 0.1
momentum = 0.9
batch_size = 256
epochs = 50
lr_decay = 1e-6
lr_drop = 30
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=196,
                decay_rate=0.9,
                staircase=True)
optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, 
                            momentum=momentum) #Define optimizer
"""
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))    
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)    
optimizer = tf.optimizers.SGD(learning_rate=learning_rate, 
                            momentum=momentum, decay = lr_decay, nesterov= True) #Define optimizer
"""

################################################################
# TESTING THE MODEL:
general_testing(isAConnect=isAConnect,
                Wstd_err=Wstd_err,
                Sim_err=Sim_err,
                WisQuant=WisQuant,BisQuant=BisQuant,
                Wbw=Bbw,Bbw=Bbw,
                Conv_pool=Conv_pool,
                errDistr=errDistr,
                namev=namev,
                optimizer=optimizer,
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                batch_size=batch_size,
                MCsims=MCsims,force=force,force_save=force_save,
                folder_models=folder_models,
                folder_results=folder_results)

