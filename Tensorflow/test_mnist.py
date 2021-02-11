import tensorflow as tf
import numpy as np
from Networks import MNIST_mismatch
from Scripts import customTraining
from Scripts import ACModel
from ACModel import ACModel
from customTraining import CustomModel
from Layers import fullyconnected
from Layers import AConnect
import mylib as my
import datetime

#tf.compat.v1.disable_eager_execution()

#print(tf.executing_eagerly())

batch_size = 256
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0
N = 5
if(N==0):
	string = "no_reg_network"
elif(N==1):
	string = "dropout_network"
elif(N==2):
	string = "dropconnect_network"
elif(N==3):
	string = "aconnect_network"
elif(N==4):
	string = "FCquant_network"	
else:
	string = "aconnect_network_bw"
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
model = MNIST_mismatch.Test_MNIST(N)
#log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#model = ACModel([
#			tf.keras.layers.Flatten(input_shape=(28,28)),
#			AConnect.AConnect(256),
#			tf.keras.layers.ReLU(),
#			AConnect.AConnect(128,0.5),
#			tf.keras.layers.ReLU(),
#			AConnect.AConnect(10),
#			tf.keras.layers.Softmax()
#])


model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = 20,batch_size=batch_size)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
## Saving the training data
np.savetxt('./Models/Training data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
np.savetxt('./Models/Training data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

model.save('./Models/'+string+'.h5',include_optimizer=True)

#my.plot_full_history(acc,val_acc,loss,val_loss,range(10))

