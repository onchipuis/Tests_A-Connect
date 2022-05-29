import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import time
from aconnect1 import layers, scripts
custom_objects = {'Conv_AConnect':layers.Conv_AConnect,'FC_AConnect':layers.FC_AConnect}

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
 X_train = np.pad(X_train,((0,0),(2,2),(2,2)), 'constant')
 X_test = np.pad(X_test,((0,0),(2,2),(2,2)), 'constant')
 X_test = np.float32(X_test) #Convert it to float32

# Get LeNet-5 Model
net='../Models/LeNet5_MNIST/2Werr_Wstd_50_Bstd_50_1bQuant_normalDistr.h5'
model = tf.keras.models.load_model(net,custom_objects=custom_objects)

# Break the model in two NNs. The idea is to obtain the input the the last FC
# layer, and the weights and biases of the last FC layer:
# Obtaining the input of the last FC layer:
model1=Model(inputs=model.input,outputs=model.layers[-3].output)
Y1 = model1.predict(X_test) # Input to last FC layer
Y1lsb = Y1.max()/16         # Quantized to 4bits
Y1int = np.round(Y1/Y1lsb)  # Integer (actual input to be saved)
Y1 = Y1int*Y1lsb            # Floating point

# Obtain new network to test if the model break-up was successful:
input_shape = model.layers[-2].get_input_shape_at(0)[1:]
xi=tf.keras.layers.Input(shape=input_shape)
x=xi
x=model.layers[-2](x)
x=model.layers[-1](x)
# Las FC layer with softmax
model2=Model(inputs=xi,outputs=x)
# Output
Y2 = model2.predict(Y1int)
print("top-1 score:",get_top_n_score(Y_test, Y2, 1))

# Save the input of the last FC layer as the new database to be used in the ML
# accelerator spice simulation.
Yaux = np.zeros(shape=[10000,121],dtype=int)# Create a matrix with 121 columns (11x11 image)
Yaux[:,0:84] = Y1int.astype('int')          # Save the database (of size 84) 
np.savetxt("Database/LeNet5_LastFC_allData.txt", Yaux, fmt='%i', delimiter =",")

# Save weights and biases of the last FC layer:
param=model.layers[-2].get_weights()
w=(np.sign(param[0]).astype(int)+1)/2
#bint=np.round(param[1]/Y1lsb).astype(int)+2**3
b=np.ones(shape=[4,10],dtype=int)
b[3,:] = np.zeros(shape=[1,10],dtype=int)

param_all = np.zeros(shape=[136,128],dtype=int)
param_all[0:84,0:10] = w
param_all[85:107,0:10] = np.ones(shape=[22,10],dtype=int)
param_all[128:132,0:10] = b
np.savetxt("Networks/LeNet5_LastLayer_Wstd_50_1bQuant_normalDistr.txt",param_all.transpose(), fmt='%i', delimiter =" ")
