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
Wstd_err = [0.1]   # Define the stddev for training
Sim_err = [0.1]
Conv_pool = [8]
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
                       
                            if not(os.path.exists(folder_results+name_sim+'.txt')) or force_save: 
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
