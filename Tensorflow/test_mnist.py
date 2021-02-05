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

#Option 1: Weights quantizied + custom backprop
#Option 2: Custom network dense layer+dropout
#Option 3: Only weights quantizied
#Option 4: Dropconnect network
#tf.compat.v1.disable_eager_execution()

#print(tf.executing_eagerly())

batch_size = 256
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model, fname_test, fname_train = MNIST_mismatch.Test_MNIST(3)
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#model = ACModel([
#			tf.keras.layers.Flatten(input_shape=(28,28)),
#			AConnect.AConnect(128,0.5),
#			tf.keras.layers.ReLU(),
#			AConnect.AConnect(10),
#			tf.keras.layers.Softmax()
#])

string = "./Models/aconnect_network.h5"
model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = 2,batch_size=batch_size,callbacks=[tensorboard_callback])
model.save(string,include_optimizer=True)

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#my.plot_full_history(acc,val_acc,loss,val_loss,range(10))

