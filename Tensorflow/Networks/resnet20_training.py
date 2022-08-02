# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
from ResNet import resnet_v1, resnet_v2
from general_training import general_training
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from aconnect1 import layers, scripts
#from aconnect import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    n = 3
    depth = n * 6 + 2
    namev = ''
elif version == 2:
    n = 2
    depth = n * 9 + 2
    namev = '_v2'

#Extra code to improve model accuracy
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

# Load the CIFAR10 data.
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = normalization(X_train,X_test)    
# Input image dimensions.
input_shape = X_train.shape[1:]

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect
Wstd_err = [0.5]   # Define the stddev for training
Conv_pool = [8]
FC_pool = [2]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = [8]
#errDistr = ["lognormal"]
errDistr = ["normal"]
saveModel = True
model_name = 'ResNet20_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
if isAConnect[0]:
    net_base = folder_models+'Wstd_0_Bstd_0.h5'
    #net_base = folder_models+'8Werr_Wstd_70_Bstd_70_8bQuant_normalDistr.h5'
    #net_base = folder_models+'8Werr_Wstd_50_Bstd_50_8bQuant_lognormalDistr.h5'
    model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)
    transferLearn = True

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
general_training(model_int=resnet_v1 if(version==1) else resnet_v2,isAConnect=isAConnect,
                        model_base=model_base,transferLearn=transferLearn,
                        Wstd_err=Wstd_err,
                        WisQuant=WisQuant,BisQuant=BisQuant,
                        Wbw=Bbw,Bbw=Bbw,
                        Conv_pool=Conv_pool,
                        FC_pool=FC_pool,
                        errDistr=errDistr,
                        input_shape=input_shape,depth=depth,namev=namev,
                        optimizer=optimizer,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        saveModel=saveModel,folder_models=folder_models,
                        folder_results=folder_results)

