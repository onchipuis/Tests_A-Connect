import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
import mylib as my
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import FC_quant
from Layers import DropConnect
from Layers import DropLayer
from Layers import fullyconnected
from Layers import dropconnect2
from Layers import AConnect

def Test_MNIST(opt):

	#Keras dense network with no regularization
	if(opt==0):

		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dense(10),			
			tf.keras.layers.Softmax()
		])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
		fname_test = 'MNIST_keras_layers'
		fname_train = 'MNIST_keras_layers_test'
		#model.save(folder+'Models/Noreg')
		return model, fname_test, fname_train
		#predictions = model.predict(x_test)	
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)
	#Keras dense network with dropout
	
	elif(opt==1):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			tf.keras.layers.Dense(128),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10),
			tf.keras.layers.Softmax()
		])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
		fname_test = 'MNIST_dropout_keras_layers'
		fname_train = 'MNIST_dropout_keras_layers_test'
		#model.save(folder+'Models/wDropo')
		return model, fname_test, fname_train
		#predictions = model.predict(x_test)
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)
	
	#Custom fullyconnected with no regularization
	
	#elif(opt==2):
	
	#	model = tf.keras.Sequential([
	#		tf.keras.layers.Flatten(input_shape=(28,28)),
	#		fullyconnected.fullyconnected(128),
	#		tf.keras.layers.ReLU(),
	#		fullyconnected.fullyconnected(10),
	#		tf.keras.layers.Softmax()
	#	])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	#	fname_test = 'MNIST_custom_fc_layer'
	#	fname_train = 'MNIST_custom_fc_layers_test'
	#	return model, fname_test, fname_train
		#predictions = model.predict(x_test)
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)		
	
	#Custom fullyconnected with dropout
	
	#elif(opt==3):
	
	#	model = tf.keras.Sequential([
	#		tf.keras.layers.Flatten(input_shape=(28,28)),
	#		fullyconnected.fullyconnected(128),
	#		tf.keras.layers.Dropout(0.5),
	#		tf.keras.layers.ReLU(),
	#		fullyconnected.fullyconnected(10),
	#		tf.keras.layers.Softmax()
	#	])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	#	fname_test = 'MNIST_dropout_custom_fc_layer'
	#	fname_train = 'MNIST_dropout_custom_fc_layer'
	#	return model, fname_test, fname_train
		#predictions = model.predict(x_test)
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)		
		
	#Custom dropout layer with keras dense
	
	#elif(opt==4):
	
	#	model = tf.keras.Sequential([
	#		tf.keras.layers.Flatten(input_shape=(28,28)),
	#		tf.keras.layers.Dense(128),
	#		DropLayer.DropLayer(0.5),
	#		tf.keras.layers.ReLU(),
	#		tf.keras.layers.Dense(10),
	#		tf.keras.layers.Softmax()
	#	])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	#	fname_test = 'MNIST_dropoutcustom_fc_layer'
	#	fname_train = 'MNIST_dropoutcustom_fc_layer'
	#	return model, fname_test, fname_train
		#predictions = model.predict(x_test)
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)		
		
	#Custom dropout layer with custom dense
	
	#elif(opt==5):
	
	#	model = tf.keras.Sequential([
	#		tf.keras.layers.Flatten(input_shape=(28,28)),
	#		fullyconnected.fullyconnected(128),
	#		DropLayer.DropLayer(0.5),
	#		tf.keras.layers.ReLU(),
	#		fullyconnected.fullyconnected(10),
	#		tf.keras.layers.Softmax()
	#	])
	#	
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
	#	fname_test = 'MNIST_custom_dropout_fc_layer'
	#	fname_train = 'MNIST_custom_dropout_fc_layer'
	#	return model, fname_test, fname_train
		#predictions = model.predict(x_test)
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)		
	
	#Custom Dropconnect layer with mismatch
	
	if(opt==2):
	
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),
			DropConnect.DropConnect(128,0.5),
			tf.keras.layers.BatchNormalization(),			
			tf.keras.layers.ReLU(),
			DropConnect.DropConnect(10),
			tf.keras.layers.Softmax()
		])
		
		#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		#history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = epochs)
		fname_test = 'MNIST_Dropconnect_layer'
		#predictions = model.predict(x_test)
		fname_train = 'MNIST_Dropconnect_layer_test'
		#model.save(folder+'/Models/wDropC')
		return model, fname_test, fname_train
		#my.plot_test_imgs(predictions,y_test,x_test, numbers, 5 , 3,fname_test)
		#my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(epochs),fname_train)
		#print("Network trained: %d" % opt)

	if(opt==3):
		
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28,28)),	
			AConnect.AConnect(128,0.5),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.ReLU(),
			AConnect.AConnect(10,0.5),
			tf.keras.layers.Softmax()
		])
		fname_test = 'MNIST_AConnect_layer'
		fname_train = 'MNIST_AConnect_layer_test'
#		model.save(folder+'Models/wAConn')
		return model, fname_test, fname_train		
