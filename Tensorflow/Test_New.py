##Prueba para extraer las pruebas para 30, 50 y 70% con/sin binarización

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

from Scripts import load_ds
from Layers import AConnect
import datetime

def Test_MNIST(imgsize=[28,28],Wstd=0,Bstd=0,isBin="no"):
	
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=imgsize),	
	   	AConnect.AConnect(128,Wstd,Bstd,isBin),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.ReLU(),
		AConnect.AConnect(10,Wstd,Bstd,isBin),
		tf.keras.layers.Softmax()
		])

	return model	

## Elegir que dataset se va a utilizar	
#Opt = input("Please select which dataset do you want to test: \n 1. MNIST 28x28 8 bits \n 2. MNIST 28x28 4 bits \n 3. MNIST 11x11 8 bits \n 4. MNIST 11x11 4 bits \n Option: ")
#Opt = int(Opt)
batch_size = 256
for Opt in range(4):
	
	if(Opt == 0): #For standard MNIST 28x28 8 bits
		imgsize = [28,28]
		Q = 8
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		Wstd = 0.5
		Bstd = 0.5
		isBin = 1
		isBin2 = 'yes'
		
		string = "aconnect_network"
		if(isBin==1):
			if(Wstd !=0 or Bstd !=0):
				string = string+'_bw'
			else:
				string='FCquant_network'
		else:
			string = string

		if(imgsize==[11,11]):
			if(Q == 4):
				string=string+'_11x11_4b'
			else:
				string = string+'_11x11_8b'
		else:
			if(Q == 4):
				string=string+'_28x28_4b'
			else:
				string = string+'_28x28_8b'

		if(Wstd !=0 or Bstd !=0):
			if(Wstd != 0):
				string = string+'_'+str(int(100*Wstd))
			if(Bstd != 0):
				string = string+'_'+str(int(100*Bstd))
	
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
		model = Test_MNIST(imgsize,Wstd,Bstd,isBin2)


		print('\n\n*******************************************************************************************\n\n')
		print('TRAINING NETWORK: ', string)
		print('\n\n*******************************************************************************************')
		model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		## Saving the training data
		np.savetxt('./Models_New/Training_data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
		np.savetxt('./Models_New/Training_data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

		string = './Models_New/'+string+'.h5'
		model.save(string,include_optimizer=True)

		options1 = [str(Opt), str(Wstd), str(Bstd), str(isBin), str(3)]
		data1 = "Train_Options1.txt"
		f1 = open(data1,"w")
		for i in options1:
			f1.write(i+'\n')
		f1.close()
		
	elif(Opt==1): #For MNIST 28x28 4 bits
		imgsize = [28,28]
		Q = 4
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		Wstd = 0.5
		Bstd = 0.5
		isBin = 1
		isBin2 = 'yes'
		
		string = "aconnect_network"
		if(isBin==1):
			if(Wstd !=0 or Bstd !=0):
				string = string+'_bw'
			else:
				string='FCquant_network'
		else:
			string = string

		if(imgsize==[11,11]):
			if(Q == 4):
				string=string+'_11x11_4b'
			else:
				string = string+'_11x11_8b'
		else:
			if(Q == 4):
				string=string+'_28x28_4b'
			else:
				string = string+'_28x28_8b'

		if(Wstd !=0 or Bstd !=0):
			if(Wstd != 0):
				string = string+'_'+str(int(100*Wstd))
			if(Bstd != 0):
				string = string+'_'+str(int(100*Bstd))
	
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
		model = Test_MNIST(imgsize,Wstd,Bstd,isBin2)


		print('\n\n*******************************************************************************************\n\n')
		print('TRAINING NETWORK: ', string)
		print('\n\n*******************************************************************************************')
		model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		## Saving the training data
		np.savetxt('./Models_New/Training_data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
		np.savetxt('./Models_New/Training_data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

		string = './Models_New/'+string+'.h5'
		model.save(string,include_optimizer=True)

		options2 = [str(Opt), str(Wstd), str(Bstd), str(isBin), str(3)]
		data2 = "Train_Options2.txt"
		f2 = open(data2,"w")
		for i in options2:
			f2.write(i+'\n')
		f2.close()
		
			
	elif(Opt==2): #For MNIST 11x11 8 bits
		imgsize = [11,11] 
		Q = 8
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		Wstd = 0.5
		Bstd = 0.5
		isBin = 1
		isBin2 = 'yes'
		
		string = "aconnect_network"
		if(isBin==1):
			if(Wstd !=0 or Bstd !=0):
				string = string+'_bw'
			else:
				string='FCquant_network'
		else:
			string = string

		if(imgsize==[11,11]):
			if(Q == 4):
				string=string+'_11x11_4b'
			else:
				string = string+'_11x11_8b'
		else:
			if(Q == 4):
				string=string+'_28x28_4b'
			else:
				string = string+'_28x28_8b'

		if(Wstd !=0 or Bstd !=0):
			if(Wstd != 0):
				string = string+'_'+str(int(100*Wstd))
			if(Bstd != 0):
				string = string+'_'+str(int(100*Bstd))
	
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
		model = Test_MNIST(imgsize,Wstd,Bstd,isBin2)


		print('\n\n*******************************************************************************************\n\n')
		print('TRAINING NETWORK: ', string)
		print('\n\n*******************************************************************************************')
		model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		## Saving the training data
		np.savetxt('./Models_New/Training_data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
		np.savetxt('./Models_New/Training_data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

		string = './Models_New/'+string+'.h5'
		model.save(string,include_optimizer=True)

		options3 = [str(Opt), str(Wstd), str(Bstd), str(isBin), str(3)]
		data3 = "Train_Options3.txt"
		f3 = open(data3,"w")
		for i in options3:
			f3.write(i+'\n')
		f3.close()
		
	elif(Opt==3): #For MNIST 11x11 4 bits
		imgsize = [11,11] 
		Q = 4
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		Wstd = 0.5
		Bstd = 0.5
		isBin = 1
		isBin2 = 'yes'
		
		string = "aconnect_network"
		if(isBin==1):
			if(Wstd !=0 or Bstd !=0):
				string = string+'_bw'
			else:
				string='FCquant_network'
		else:
			string = string

		if(imgsize==[11,11]):
			if(Q == 4):
				string=string+'_11x11_4b'
			else:
				string = string+'_11x11_8b'
		else:
			if(Q == 4):
				string=string+'_28x28_4b'
			else:
				string = string+'_28x28_8b'

		if(Wstd !=0 or Bstd !=0):
			if(Wstd != 0):
				string = string+'_'+str(int(100*Wstd))
			if(Bstd != 0):
				string = string+'_'+str(int(100*Bstd))
	
		optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
		model = Test_MNIST(imgsize,Wstd,Bstd,isBin2)


		print('\n\n*******************************************************************************************\n\n')
		print('TRAINING NETWORK: ', string)
		print('\n\n*******************************************************************************************')
		model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
		history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		## Saving the training data
		np.savetxt('./Models_New/Training_data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
		np.savetxt('./Models_New/Training_data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

		string = './Models_New/'+string+'.h5'
		model.save(string,include_optimizer=True)

		options4 = [str(Opt), str(Wstd), str(Bstd), str(isBin), str(3)]
		data4= "Train_Options4.txt"
		f4 = open(data4,"w")
		for i in options4:
			f4.write(i+'\n')
		f4.close()		 
	else:
		print ("Wrong option")
#### Load dataset  from above options
"""(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)


##### Define your Weights and biases standard deviation for training
Wstd= input("Please define the weights standard deviation for training: ")
Bstd= input("Please define the bias standard deviation for training: ")

Wstd = float(Wstd)
Bstd = float(Bstd)

#### Do you want binary weights?
isBin = 'yes'#input("Do you want binary weights? yes or no: ")
if(isBin == 'yes'):
	isBin = 1
	isBin2 = 'yes'
else:
	isBin = 0
	isBin2 = 'no'	
	

string = "aconnect_network"
	if(isBin==1):
		if(Wstd !=0 or Bstd !=0):
			string = string+'_bw'
		else:
			string='FCquant_network'
	else:
		string = string

if(imgsize==[11,11]):
	if(Q == 4):
		string=string+'_11x11_4b'
	else:
		string = string+'_11x11_8b'
else:
	if(Q == 4):
		string=string+'_28x28_4b'
	else:
		string = string+'_28x28_8b'

if(Wstd !=0 or Bstd !=0):
	if(Wstd != 0):
		string = string+'_'+str(int(100*Wstd))
	if(Bstd != 0):
		string = string+'_'+str(int(100*Bstd))
	
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
model = Test_MNIST(imgsize,Wstd,Bstd,isBin2)


print('\n\n*******************************************************************************************\n\n')
print('TRAINING NETWORK: ', string)
print('\n\n*******************************************************************************************')
model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
## Saving the training data
np.savetxt('./Models_New/Training_data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
np.savetxt('./Models_New/Training_data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

string = './Models_New/'+string+'.h5'
model.save(string,include_optimizer=True)

options = [str(Opt), str(Wstd), str(Bstd), str(isBin)]
data = "Train_Options.txt"
f = open(data,"w")
for i in options:
	f.write(i+'\n')
f.close() """

