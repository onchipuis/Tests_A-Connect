import numpy as np
import tensorflow as tf
import CNN_fashion as fashion
from general_training import general_training
from tensorflow.keras.datasets import fashion_mnist

#Extra code to improve model accuracy
def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2))
    std = np.std(train_images, axis=(0, 1, 2))
    train_images =(train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

# LOADING DATASET:
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train, X_test = normalization(X_train,X_test)    
X_train = np.pad(X_train, ((0,0),(2,2),(2,2)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2)), 'constant')

# INPUT PARAMTERS:
isAConnect = [True]   # Which network you want to train/test True for A-Connect false for normal LeNet
#Wstd_err = [0,0.3,0.5,0.7]   # Define the stddev for training
Wstd_err = [0.7]
Conv_pool = [8]
FC_pool = Conv_pool
WisQuant = ["yes"]		    # Do you want binary weights?
BisQuant = WisQuant 
Wbw = [8]
Bbw = [8]
saveModel = True
#errDistr = ["lognormal"]
errDistr = ["normal"]
model_name = 'CNN_FASHION_MNIST/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'

# TRAINING PARAMETERS
learning_rate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum) #Define optimizer
batch_size = 256
epochs = 50

# TRAINING THE MODEL:
general_training(model_int=fashion.model_creation,isAConnect=isAConnect,
                        model_base=None,transferLearn=False,
                        Wstd_err=Wstd_err,
                        WisQuant=WisQuant,BisQuant=BisQuant,
                        Wbw=Bbw,Bbw=Bbw,
                        Conv_pool=Conv_pool,
                        FC_pool=FC_pool,
                        errDistr=errDistr,
                        input_shape=None,depth=None,namev='',
                        optimizer=optimizer,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=None,
                        saveModel=saveModel,folder_models=folder_models,
                        folder_results=folder_results)

"""
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
    
    for i in range(len(FC_pool_aux)):
        for p in range (len(WisQuant_aux)):
            if WisQuant[p]=="yes":
                Wbw_aux = Wbw
                Bbw_aux = Bbw
            else:
                Wbw_aux = [8]
                Bbw_aux = [8]

            for q in range (len(Wbw_aux)):
                for j in range(len(Wstd_aux)): #Iterate over the Wstd and Bstd for training
                    for k in range(len(errDistr_aux)):
                        Err = Wstd_aux[j]
                        # CREATING NN:
                        model = lenet5.model_creation(isAConnect=isAConnect[d],
                                                        Wstd=Err,Bstd=Err,
                                                        isQuant=[WisQuant_aux[p],BisQuant_aux[p]],
                                                        bw=[Wbw_aux[q],Bbw_aux[q]],
                                                        Conv_pool=Conv_pool_aux[i],
                                                        FC_pool=FC_pool_aux[i],
                                                        errDistr=errDistr_aux[k])
                        # NAME
                        if isAConnect[d]:
                            Werr = str(int(100*Err))
                            Nm = str(int(FC_pool_aux[i]))
                            if WisQuant_aux[p] == "yes":
                                bws = str(int(Wbw_aux[q]))
                                quant = bws+'bQuant_'
                            else:
                                quant = ''
                            
                            if Werr == '0':
                                name = 'Wstd_'+Werr+'_Bstd_'+Werr
                            else:
                                name = Nm+'Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+quant+errDistr_aux[k]+'Distr'
                        
                        else:
                            name = 'Base'

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
"""
