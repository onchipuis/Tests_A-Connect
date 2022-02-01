import numpy as np
import tensorflow as tf
import LeNet5 as lenet5
import time
from aconnect1 import layers, scripts
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
(X_train, Y_train), (X_test, Y_test) = scripts.load_ds() #Load dataset
X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.float32(X_test) #Convert it to float32

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
#Wstd_err = [0.3,0.5,0.7]   # Define the stddev for training
Wstd_err = [0.3]	    # Define the stddev for training
Conv_pool = [2]
FC_pool = Conv_pool
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [2]
Bbw = Wbw
#errDistr = "lognormal"
saveModel = False
errDistr = ["normal"]
model_name = 'LeNet5_MNIST/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'

# TRAINING PARAMETERS
learning_rate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum) #Define optimizer
batch_size = 256
epochs = 20

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
        for p in range (len(WisQuant)):
            if WisQuant[p]=="yes":
                Wbw_aux = Wbw
                Bbw_aux = Bbw
            else:
                Wbw_aux = [8]
                Bbw_aux = [8]

            for q in range (len(Wbw_aux)):
                for j in range(len(Wstd_aux)): #Iterate over the Wstd and Bstd for training
                    for k in range(len(errDistr)):
                        Err = Wstd_aux[j]
                        # CREATING NN:
                        model = lenet5.model_creation(isAConnect=isAConnect[d],
                                                        Wstd=Err,Bstd=Err,
                                                        isQuant=[WisQuant[p],BisQuant[p]],
                                                        bw=[Wbw_aux[q],Bbw_aux[q]],
                                                        Conv_pool=Conv_pool_aux[i],
                                                        FC_pool=FC_pool_aux[i],
                                                        errDistr=errDistr[k])
                        Werr = str(int(100*Err))
                        Nm = str(int(FC_pool_aux[i]))
                        if WisQuant[p] == "yes":
                            bws = str(int(Wbw_aux[q]))
                            quant = bws+'bQuant_'
                        else:
                            quant = ''
                        name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr[k]+'Distr'

                        print("*************************TRAINING NETWORK*********************")
                        print("\n\t\t\t", name)
                        
                        #TRAINING PARAMETERS
                        model.compile(optimizer=optimizer,
                                    loss=['sparse_categorical_crossentropy'],
                                    metrics=['accuracy'])#Compile the model
                        
                        # TRAINING
                        history = model.fit(X_train,Y_train,
                                            batch_size=batch_size,
                                            epochs = epochs,
                                            validation_data=(X_test, Y_test),
                                            shuffle=True)
                        model.evaluate(X_test,Y_test)    

                        Y_predict =model.predict(X_test)
                        elapsed_time = time.time() - start_time
                        print("top-1 score:", get_top_n_score(Y_test, Y_predict, 1))
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
                            np.savetxt(folder_results+name+'_acc'+'.txt',acc,fmt="%.2f") 
                            np.savetxt(folder_results+name+'_val_acc'+'.txt',val_acc,fmt="%.2f")
