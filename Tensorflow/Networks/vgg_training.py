"""
Script for training VGG with A-Connect, DVA, or none (Baseline)
INSTRUCTIONS:
Due to the memory usage we recommend to uncomment the first train the model and save it. Then just comment the training stage and then load the model to test it using the Monte Carlo simulation.
"""
import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')

import numpy as np
import tensorflow as tf
import VGG as vgg
import time
#from keras.callbacks import LearningRateScheduler
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
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
# CREATING NN:
model = vgg.model_creation(isAConnect=False,Wstd=0,Bstd=0)

##### PRETRAINED WEIGHTS FOR HIGHER ACCURACY LEVELS
model_aux=tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                                       input_shape=(32,32,3))

# Without Aconnect
model.layers[1].set_weights(model_aux.layers[1].get_weights())
model.layers[2].set_weights(model_aux.layers[2].get_weights())
model.layers[4].set_weights(model_aux.layers[3].get_weights())
model.layers[7].set_weights(model_aux.layers[5].get_weights())
model.layers[9].set_weights(model_aux.layers[6].get_weights())
model.layers[12].set_weights(model_aux.layers[8].get_weights())
model.layers[14].set_weights(model_aux.layers[9].get_weights())
model.layers[16].set_weights(model_aux.layers[10].get_weights())
model.layers[19].set_weights(model_aux.layers[12].get_weights())
model.layers[21].set_weights(model_aux.layers[13].get_weights())
model.layers[23].set_weights(model_aux.layers[14].get_weights())
model.layers[26].set_weights(model_aux.layers[16].get_weights())
model.layers[28].set_weights(model_aux.layers[17].get_weights())
model.layers[30].set_weights(model_aux.layers[18].get_weights())

## With Aconnect
"""
model.layers[1].set_weights(model_aux.layers[1].get_weights())
model.layers[4].set_weights(model_aux.layers[2].get_weights())
model.layers[8].set_weights(model_aux.layers[4].get_weights())
model.layers[11].set_weights(model_aux.layers[5].get_weights())
model.layers[15].set_weights(model_aux.layers[7].get_weights())
model.layers[18].set_weights(model_aux.layers[8].get_weights())
model.layers[21].set_weights(model_aux.layers[9].get_weights())
model.layers[25].set_weights(model_aux.layers[11].get_weights())
model.layers[28].set_weights(model_aux.layers[12].get_weights())
model.layers[31].set_weights(model_aux.layers[13].get_weights())
model.layers[35].set_weights(model_aux.layers[15].get_weights())
model.layers[38].set_weights(model_aux.layers[16].get_weights())
model.layers[41].set_weights(model_aux.layers[17].get_weights())
"""

print(model.summary())

#TRAINING PARAMETERS
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

# TRAINING
model.fit(train_images, train_labels,
          batch_size=256,epochs=30,
          validation_data=(test_images,test_labels),
          )
model.evaluate(test_images,test_labels)    

y_predict =model.predict(test_images)
elapsed_time = time.time() - start_time
print("top-1 score:", get_top_n_score(test_labels, y_predict, 1))
print("Elapsed time: {}".format(hms_string(elapsed_time)))
print('Tiempo de procesamiento (secs): ', time.time()-tic)

#model.save("./Models/CifarVGG_Aconnect05.h5",include_optimizer=True) ### MODEL SAVING LOGIC
