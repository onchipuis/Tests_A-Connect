import Networks
import numpy as np
import mylib as my
import tensorflow as tf
from Networks import MNIST_mismatch

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

numbers = ['0','1','2','3','4','5','6','7','8','9']
epochs = 10
fname_test = ['','','','','','','','']
fname_train = ['','','','','','','','']
model = [[],[],[],[]]
predictions = [[],[],[],[]]
history = [[],[],[],[]]
for i in range(4):
	temp, fname_test[i], fname_train[i] = MNIST_mismatch.Test_MNIST(i)
	model[i] = temp
		
for j in range(4):
	print("Train network: %d" % j)
	model[j].compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
	history[j] = model[j].fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	predictions[j] = model[j].predict(x_test)
	my.plot_test_imgs(predictions[j],y_test,x_test, numbers, 5 , 3,fname_test[j])
	my.plot_full_history(history[j].history['accuracy'],history[j].history['val_accuracy'],history[j].history['loss'],history[j].history['val_loss'],range(epochs),fname_train[j])	

