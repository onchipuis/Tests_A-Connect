# Based on https://keras.io/zh/examples/cifar10_resnet/
import tensorflow as tf
import numpy as np
import os
import time
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
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# INPUT PARAMTERS:
saveModel = True
model_name = 'ResNet20_CIFAR10/'
folder_models = './Models/'+model_name
folder_results = '../Results/'+model_name+'Training_data/'
#net_base = folder_models+'2Werr_Wstd_50_Bstd_50_8bQuant_normalDistr.h5'
net_base = folder_models+'Base.h5'
Wstd = [0.3,0.5,0.7]   # Define the stddev for training

# Load the CIFAR10 data.
#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Input image dimensions.
#input_shape = X_train.shape[1:]
################################################################
### TRAINING
for j in range(len(Wstd)):

    Werr = str(int(100*Wstd[j]))
    name = '8Werr'+'_Wstd_'+Werr+'_Bstd_'+Werr+'_'+'8bQuant_lognormalDistr.h5'
    model = tf.keras.models.load_model(folder_models+name,custom_objects=custom_objects)
    w = np.array(model.get_weights(),dtype=object)
    model.set_weights(np.multiply(w,np.exp(0.5*np.power(Wstd[j],2))))
    model.save(name,include_optimizer=False)
