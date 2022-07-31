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


"""
for d in range(len(isAConnect)): #Iterate over the networks
    if isAConnect[d]: #is a network with A-Connect?
        Wstd_aux = Wstd_err
        Conv_pool_aux = Conv_pool
    else:
        Wstd_aux = [0]
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
                        for m in range(len(Sim_err)):

                            Werr = Wstd_aux[j]
                            Err = Sim_err[m]
                            # NAME
                            if isAConnect[d]:
                                Werr = str(int(100*Werr))
                                Nm = str(int(Conv_pool_aux[i]))
                                if WisQuant[p] == "yes":
                                    bws = str(int(Wbw_aux[q]))
                                    quant = bws+'bQuant_'
                                else:
                                    quant = ''
                                if Werr == '0':
                                    name = 'Wstd_'+Werr+'_Bstd_'+Werr
                                else:
                                    name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr[k]+'Distr'

                            else:
                                name = 'Base'
                            string = folder_models + name + '.h5'
                            name_sim = name+'_simErr_'+str(int(100*Err))                      
                            name_stats = name+'_stats_simErr_'+str(int(100*Err))                      
                       
                            if not os.path.exists(folder_results+name_sim+'.txt'): 
                            #if os.path.exists(folder_results+name_sim+'.txt'): 
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
"""
