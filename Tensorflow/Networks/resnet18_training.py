# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
from ResNet import resnet_v1, resnet_v2
from general_training import general_training
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar100
from aconnect import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

#Extra code to improve model accuracy
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

# Load the CIFAR100 data.
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
X_train, X_test = normalization(X_train,X_test)    
# Input image dimensions.
input_shape = X_train.shape[1:]

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect
Wstd_err = [0]   # Define the stddev for training
Conv_pool = [1]
FC_pool = [1]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = [8]
#errDistr = ["lognormal"]
errDistr = ["normal"]
saveModel = True
model_name = 'ResNet18_CIFAR100/'
folder_models = './Models/'+model_name
#if isAConnect[0]:
#    net_base = folder_models+'Wstd_0_Bstd_0_8bQuant.h5'
    #net_base = folder_models+'8Werr_Wstd_70_Bstd_70_8bQuant_normalDistr.h5'
    #net_base = folder_models+'8Werr_Wstd_50_Bstd_50_8bQuant_lognormalDistr.h5'
#    model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)
#    transferLearn = True

# Does include error matrices during backward propagation?
bwErrProp = [True]
if not(bwErrProp[0]):
    model_name = model_name+'ForwNoise_only/' 
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'

# TRAINING PARAMETERS
#lrate = 1e-3        # for Adam optimizer
if isAConnect[0]:
    lrate = 1e-1
    epochs = 120
    epoch1 = 30
    epoch2 = 60
    epoch3 = 100
    #epochs = 60
    #epoch1 = 25
    #epoch2 = 40
    #epoch3 = 50
    #epoch1 = 60
    #epoch2 = 90
    #epoch3 = 120
else:
    lrate = 1e-1
    epochs = 200
    epoch1 = 80
    epoch2 = 120
    epoch3 = 160
num_classes = 10
momentum = 0.9
batch_size = 256
lr_decay = 0#1e-4
lr_drop = 20

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = lrate
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > epoch3:
        lr *= 1e-3
    elif epoch > epoch2:
        lr *= 1e-2
    elif epoch > epoch1:
        lr *= 1e-1
    
    print('Learning rate: ', lr)
    return lr

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0)
optimizer = tf.optimizers.SGD(learning_rate=0.0, 
                            momentum=momentum, nesterov= True, decay = lr_decay) #Define optimizer

################################################################
# TRAINING THE MODEL:
general_training(model_int=resnet18,isAConnect=True,
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

