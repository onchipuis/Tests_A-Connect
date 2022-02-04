"""
Script for training VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import numpy as np
import tensorflow as tf
import VGG1 as vgg
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from aconnect1 import layers, scripts
#from aconnect import layers, scripts
#from keras.callbacks import LearningRateScheduler
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

tic=time.time()
start_time = time.time()
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
Wstd_err = [0.7]   # Define the stddev for training
Conv_pool = [1,2,4,8,16]
FC_pool = [1,2,4,4,4]
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = Wbw
#errDistr = "lognormal"
errDistr = ["normal"]
saveModel = True
model_name = 'VGG16_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
net_base = folder_models+'Base.h5'

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
        
    for i in range(len(Conv_pool_aux)):
        for p in range (len(WisQuant)):
            if WisQuant[p]=="yes":
                Wbw_aux = Wbw
                Bbw_aux = Bbw
            else:
                Wbw_aux = [8]
                Bbw_aux = [8]

            for q in range (len(Wbw_aux)):
                for j in range(len(Wstd_aux)):
                    for k in range(len(errDistr)):
                        Err = Wstd_aux[j]
                        # CREATING NN:
                        model = vgg.model_creation(isAConnect=isAConnect[d],
                                                    Wstd=Err,Bstd=Err,
                                                    isQuant=[WisQuant[p],BisQuant[p]],
                                                    Conv_pool=Conv_pool_aux[i],
                                                    FC_pool=FC_pool_aux[i],
                                                    errDistr=errDistr[k],isBin="no")
                       
                        # NAME
                        if isAConnect[d]:
                            Werr = str(int(100*Err))
                            Nm = str(int(Conv_pool_aux[i]))
                            if WisQuant[p] == "yes":
                                bws = str(int(Wbw_aux[q]))
                                quant = bws+'bQuant_'
                            else:
                                quant = ''
                            name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr[k]+'Distr'
                        else:
                            name = 'Base'
                           
                        string = folder_models + name + '.h5'
                        model_old = tf.keras.models.load_model(string,custom_objects = custom_objects)
                        model.set_weights(model_old.get_weights())
                        

                        # SAVE MODEL:
                        if saveModel:
                            model.save(string,include_optimizer=False)
