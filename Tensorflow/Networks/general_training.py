# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
import os
import time
from aconnect import layers, scripts
#from aconnect import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

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

################################################################
### TRAINING
def general_training (model_int=None,isAConnect=[True],
                        model_base=None,transferLearn=False,
                        Wstd_err=[0],
                        WisQuant=["no"],BisQuant=["no"],
                        Wbw=[8],Bbw=[8],
                        Conv_pool=[2],
                        FC_pool=[2],
                        errDistr=["normal"],
                        bwErrProp=[True],
                        input_shape=None,depth=None,namev='', # Use for ResNet only
                        optimizer=None,
                        X_train=None, Y_train=None,
                        X_test=None, Y_test=None,
                        batch_size=256,
                        epochs=100,
                        callbacks=None,
                        saveModel=False,folder_models=None,
                        folder_results=None,
                        **kwargs):

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
                    for b in range (len(bwErrProp)):
                        for j in range(len(Wstd_aux)):
                            for k in range(len(errDistr_aux)):
                                Err = Wstd_aux[j]
                                # CREATING NN:
                                model = model_int(isAConnect = isAConnect[d], 
                                                Wstd=Err,Bstd=Err,
                                                isQuant=[WisQuant_aux[p],BisQuant_aux[p]],
                                                bw=[Wbw_aux[q],Bbw_aux[q]],
                                                Conv_pool=Conv_pool_aux[i],
                                                FC_pool=FC_pool_aux[i],
                                                errDistr=errDistr_aux[k],
                                                bwErrProp=bwErrProp[b],
                                                input_shape=input_shape,
                                                depth=depth,**kwargs)
                                
                                ##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
                                if isAConnect[d] and transferLearn:
                                    model.set_weights(model_base.get_weights())
                                
                                # NAME
                                if isAConnect[d]:
                                    Werr = str(int(100*Err))
                                    Nm = str(int(Conv_pool_aux[i]))
                                    if WisQuant_aux[p] == "yes":
                                        bws = str(int(Wbw_aux[q]))
                                        quant = bws+'bQuant'
                                    else:
                                        quant = ''
                                    if Werr == '0':
                                        name = 'Wstd_0_Bstd_0_'+quant
                                    else:
                                        name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+'_'+errDistr_aux[k]+'Distr'+namev
                                else:
                                    name = 'Base'+namev
                                
                                print("*************************TRAINING NETWORK*********************")
                                print("\n\t\t\t", folder_models + name)
                                
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
