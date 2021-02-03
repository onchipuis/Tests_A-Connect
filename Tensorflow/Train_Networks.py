import Networks
import numpy as np
import mylib as my
import tensorflow as tf
from datetime import datetime
from Networks import MNIST_mismatch
from multiprocessing import Pool



def parallel_training(n):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = x_train/255.0
	x_test = x_test/255.0
	i = n
	j = n
	numbers = ['0','1','2','3','4','5','6','7','8','9']
	epochs = 10
	fname_test = ['','','','','','','','']
	fname_train = ['','','','','','','','']
	model = [[],[],[],[]]
	predictions = [[],[],[],[]]
	history = [[],[],[],[]]
	temp, fname_test[i], fname_train[i] = MNIST_mismatch.Test_MNIST(i)
	model[i] = temp
			
	print("Train network: %d" % j)
	model[j].compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
	history[j] = model[j].fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	if j == 0:
		string = "./Models/no_reg_network.h5"
	elif j == 1:
		string = "./Models/dropout_network.h5"
	elif j == 2:
		string = "./Models/dropconnect_network.h5"
	elif j == 3:
		string = "./Models/aconnect_network.h5"
	model[j].save(string,include_optimizer=True)
pool = Pool()
now = datetime.now()
starttime = now.time()
print('\n\n*******************************************************************************************')
pool.map(parallel_training, range(4))
pool.close()
now = datetime.now()
endtime = now.time()
print('\n\n*******************************************************************************************')
print('\n\nTraining started at: ',starttime)
print('Training finished at: ', endtime)
#print('\n\n Time taken: %f' % (endtime-starttime))


