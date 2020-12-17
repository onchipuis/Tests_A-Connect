import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mylib as my
import sys
#sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/')
sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/Layers/')
sys.path.append('/home/rvergel/Desktop/Library_AConnect_TG/Scripts/')
from Scripts import CustomBackprop
from Layers import FC_quant
from Layers import DropConnect
from Layers import DropLayer
def mnist_test(opt=2):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = x_train/255.0
	x_test = x_test/255.0

	numbers = ['0','1','2','3','4','5','6','7','8','9']
	
	if(opt==1):
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape = (28,28)),
			FC_quant(128),
			tf.keras.layers.ReLU(),
			FC_quant(10),
			tf.keras.layers.Softmax()
		])
		model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15)
	elif(opt==2):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape = (28,28)),
			tf.keras.layers.Dense(128),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Softmax()
		])
		model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15)
		
	elif(opt==3):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape = (28, 28)),
			tf.keras.layers.Dense(128),
			DropLayer.DropLayer(0.5),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),
			DropLayer.DropLayer(0.5),
			tf.keras.layers.Softmax()
		])
		
		model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15)
		
	elif(opt==4):
		isBin = input("Binarized weights, yes/no: ")
		if(isBin=="yes"):
			string0 = 'MNIST_dropconnect_missmatch_binarized'
			string = 'MNIST_dropconnect_mismatch_train_binarized'
		else:
			string0 = 'MNIST_dropconnect_missmatch'
			string = 'MNIST_dropconnect_mismatch_train'
			
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape = (28,28)),
			DropConnect.DropConnect(128,0.5, isBin = isBin),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			DropConnect.DropConnect(10,0.5),
			tf.keras.layers.Softmax()
		])
		model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=15)
		
	else:
		print("Wrong option")	

	predictions = model.predict(x_test)
	if(opt==1):
		my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,'MNIST_binarized')
		my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(15),'MNIST_binarized_train')
	elif(opt==2):
		my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,'MNIST_dropout')
		my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(15),'MNIST_dropout_train')
	
	elif(opt==3):
		my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,'MNIST_custom_dropout')
		my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(15),'MNIST_custom_dropout,train')
	
	elif(opt==4):
		my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3, string0)
		my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(15),string)
	
	else:
		print("Wrong option")
	test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
	
	return test_loss, test_acc
