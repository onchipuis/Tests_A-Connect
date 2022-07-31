# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
import os
import gc
import time
from datetime import datetime
from aconnect1 import layers, scripts
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
acc=np.zeros([500,1])
def general_training (model_int=None,isAConnect=[True],
                        Wstd_err=[0],
                        WisQuant=["no"],BisQuant=["no"],
                        Wbw=[8],Bbw=[8],
                        Conv_pool=[2],
                        FC_pool=[2],
                        errDistr=["normal"],
                        namev='', # Use for ResNet only
                        optimizer=None,
                        X_train=None, Y_train=None,
                        X_test=None, Y_test=None,
                        batch_size=256,
                        MCsims=100,force="yes",force_save=True,
                        folder_models=None,
                        folder_results=None,
                        **kwargs):

    for d in range(len(isAConnect)): #Iterate over the networks
        if isAConnect[d]: #is a network with A-Connect?
            Wstd_aux = Wstd_err
            FC_pool_aux = FC_pool
            Conv_pool_aux = Conv_pool
            WisQuant_aux = WisQuant
            BisQuant_aux = BisQuant
        else:
            Wstd_aux = [0]
            FC_pool_aux = [0]
            Conv_pool_aux = [0]
            WisQuant_aux = ["no"]
            BisQuant_aux = ["no"]
            
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
                        for k in range(len(errDistr)):
                            for m in range(len(Sim_err)):
                            
                            Werr = Wstd_aux[j]
                            Err = Sim_err[m]
                            # NAME
                            if isAConnect[d]:
                                Werr = str(int(100*Werr))
                                Nm = str(int(Conv_pool_aux[i]))
                                if WisQuant_aux[p] == "yes":
                                    bws = str(int(Wbw_aux[q]))
                                    quant = bws+'bQuant_'
                                else:
                                    quant = ''
                                if Werr == '0':
                                    name = 'Wstd_0_Bstd_0'
                                else:
                                    name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr[k]+'Distr'+namev
                            else:
                                name = 'Base'+namev
                            
                            string = folder_models + name + '.h5'
                            name_sim = name+'_simErr_'+str(int(100*Err))                      
                            name_stats = name+'_stats_simErr_'+str(int(100*Err))                      
                            
                            if not(os.path.exists(folder_results+name_sim+'.txt')) or force_save: 
                                if Err == 0:
                                    N = 1
                                else:
                                    N = MCsims
                                        #####

                                elapsed_time = time.time() - start_time
                                print("Elapsed time: {}".format(hms_string(elapsed_time)))
                                now = datetime.now()
                                starttime = now.time()
                                print('\n\n******************************************************************\n\n')
                                print('TESTING NETWORK: ', name)
                                print('With simulation error: ', Err)
                                print('\n\n**********************************************************************')
                                
                                #Load the trained model
                                #net = tf.keras.models.load_model(string,custom_objects = custom_objects) 
                                net = string
                                #MC sim
                                acc, stats = scripts.MonteCarlo(net=net,Xtest=X_test,Ytest=Y_test,M=N,
                                        Wstd=Err,Bstd=Err,force=force,Derr=0,net_name=name,
                                        custom_objects=custom_objects,
                                        optimizer=optimizer,
                                        loss='sparse_categorical_crossentropy',
                                        metrics=['accuracy'],top5=False,dtype='float16',
                                        errDistr=errDistr[k],evaluate_batch_size=batch_size
                                        )
                                np.savetxt(folder_results+name_sim+'.txt',acc,fmt="%.4f")
                                np.savetxt(folder_results+name_stats+'.txt',stats,fmt="%.4f")

                                now = datetime.now()
                                endtime = now.time()
                                elapsed_time = time.time() - start_time
                                print("Elapsed time: {}".format(hms_string(elapsed_time)))

                                print('\n\n*********************************************************************')
                                print('\n Simulation started at: ',starttime)
                                print('Simulation finished at: ', endtime)
                                del net,acc,stats
                                gc.collect()
                                tf.keras.backend.clear_session()
                                tf.compat.v1.reset_default_graph()
                                #exit()
