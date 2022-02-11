# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from ResNet import resnet_v1, resnet_v2
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
tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"
#Extra code to improve model accuracy
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

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

# Load the CIFAR10 data.
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = normalization(X_train,X_test)    
# Input image dimensions.
input_shape = X_train.shape[1:]


# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.7]   # Define the stddev for training
Conv_pool = [8]
FC_pool = [2]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = Wbw
errDistr = ["lognormal"]
#errDistr = ["normal"]
saveModel = True
model_name = 'ResNet20_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
if isAConnect[0]:
    #net_base = folder_models+'Base'+namev+'.h5'
    net_base = folder_models+'8Werr_Wstd_70_Bstd_70_8bQuant_normalDistr.h5'
    #net_base = folder_models+'8Werr_Wstd_50_Bstd_50_8bQuant_lognormalDistr.h5'
    model_base = tf.keras.models.load_model(net_base,custom_objects=custom_objects)

# TRAINING PARAMETERS
lrate = 1e-2
#lrate = 1e-3        # for Adam optimizer
if isAConnect[0]:
    #epochs = 120
    epochs = 60
    epoch1 = 30
    epoch2 = 60
    epoch3 = 100
    #epoch1 = 60
    #epoch2 = 90
    #epoch3 = 120
else:
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
### TRAINING
for d in range(len(isAConnect)): #Iterate over the networks
    if isAConnect[d]: #is a network with A-Connect?
        Wstd_aux = Wstd_err
        FC_pool_aux = FC_pool
        Conv_pool_aux = Conv_pool
        WisQuant_aux = WisQuant
        BisQuant_aux = BisQuant
        errDistr_aux = errDistr
    else:
        Wstd_aux = [0]
        FC_pool_aux = [0]
        Conv_pool_aux = [0]
        WisQuant_aux = ["no"]
        BisQuant_aux = ["no"]
        errDistr_aux = ["normal"]
        
    for i in range(len(Conv_pool_aux)):
        for p in range (len(WisQuant_aux)):
            if WisQuant_aux[p]=="yes":
                Wbw_aux = Wbw
                Bbw_aux = Bbw
            else:
                Wbw_aux = [8]
                Bbw_aux = [8]

            for q in range (len(Wbw_aux)):
                for j in range(len(Wstd_aux)):
                    for k in range(len(errDistr_aux)):
                        Err = Wstd_aux[j]
                        # CREATING NN:
                        if version == 2:
                            model = resnet_v2(input_shape=input_shape, depth=depth,
                                            isAConnect = isAConnect[d], 
                                            Wstd=Err,Bstd=Err,
                                            isQuant=[WisQuant_aux[p],BisQuant_aux[p]],
                                            bw=[Wbw_aux[q],Bbw_aux[q]],
                                            Conv_pool=Conv_pool_aux[i],
                                            FC_pool=FC_pool_aux[i],
                                            errDistr=errDistr_aux[k])
                        else:
                            model = resnet_v1(input_shape=input_shape, depth=depth, 
                                            isAConnect = isAConnect[d], 
                                            Wstd=Err,Bstd=Err,
                                            isQuant=[WisQuant_aux[p],BisQuant_aux[p]],
                                            bw=[Wbw_aux[q],Bbw_aux[q]],
                                            Conv_pool=Conv_pool_aux[i],
                                            FC_pool=FC_pool_aux[i],
                                            errDistr=errDistr_aux[k])
                        
                        ##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
                        if isAConnect[d]:
                            model.set_weights(model_base.get_weights())
                        
                        # NAME
                        if isAConnect[d]:
                            Werr = str(int(100*Err))
                            Nm = str(int(Conv_pool_aux[i]))
                            if WisQuant_aux[p] == "yes":
                                bws = str(int(Wbw_aux[q]))
                                quant = bws+'bQuant_'
                            else:
                                quant = ''
                            name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr_aux[k]+'Distr'+namev
                        else:
                            name = 'Base'+namev
                        
                        print("*************************TRAINING NETWORK*********************")
                        print("\n\t\t\t", name)
                        
                        #TRAINING PARAMETERS
                        model.compile(loss='sparse_categorical_crossentropy',
                                      optimizer=optimizer,
                                      metrics=['accuracy'])

                        # Run training, with or without data augmentation.
                        history = model.fit(X_train, Y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=(X_test, Y_test),
                                      shuffle=True,
                                      callbacks=callbacks)
                        model.evaluate(X_test,Y_test) 
                        string = folder_models + name + '.h5'                                
                        model.save(string,include_optimizer=False)
                        y_predict =model.predict(X_test)
                        elapsed_time = time.time() - start_time
                        print("top-1 score:", get_top_n_score(Y_test, y_predict, 1))
                        print("Elapsed time: {}".format(hms_string(elapsed_time)))
                        print('Tiempo de procesamiento (secs): ', time.time()-tic)
                        #Save the accuracy and the validation accuracy
                        acc = history.history['accuracy'] 
                        val_acc = history.history['val_accuracy']
                                      
                        # SAVE MODEL:
                        if saveModel:
                            string = folder_models + name + '.h5'
                            model.save(string,include_optimizer=False)
                            #Save in a txt the accuracy and the validation accuracy for further analysis
                            if not os.path.isdir(folder_results):
                                os.makedirs(folder_results)
                            np.savetxt(folder_results+name+'_acc'+'.txt',acc,fmt="%.4f") 
                            np.savetxt(folder_results+name+'_val_acc'+'.txt',val_acc,fmt="%.4f")              
