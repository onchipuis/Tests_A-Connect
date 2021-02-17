import tensorflow as tf
import numpy as np
from Networks import MNIST_mismatch
from Scripts import customTraining
from Scripts import ACModel
from ACModel import ACModel
from Scripts import load_ds
from customTraining import CustomModel
from Layers import fullyconnected
from Layers import AConnect
import mylib as my
import datetime

##### Test options configuration

print("Please follow the intructions to configure the dataset you want to use\n")

Opt = input("Please select which dataset do you want to test: \n 1. MNIST 28x28 8 bits \n 2. MNIST 28x28 4 bits \n 3. MNIST 11x11 8 bits \n 4. MNIST 11x11 4 bits \n Option: ")
Opt = int(Opt)
batch_size = 256
if(Opt == 1): #For standard MNIST 28x28 8 bits
	imgsize = [28,28]
	Q = 8
elif(Opt==2): #For MNIST 28x28 4 bits
	imgsize = [28,28]
	Q = 4	
elif(Opt==3): #For MNIST 11x11 8 bits
	imgsize = [11,11] 
	Q = 8
elif(Opt==4): #For MNIST 11x11 4 bits
	imgsize = [11,11] 
	Q = 4
else:
	print ("Wrong option")
#### Load dataset  from above options
(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)


##### Normalize dataset
normalize = 'no' #DO you want to normalize the input data?
if(normalize == 'yes'):
	if(Q==8):
		x_train = x_train/255
		x_test = x_test/255
	elif(Q==4):
		x_train = x_train/15
		x_test = x_test/15
	else:
		print ("Not supported matrix quantization")

##### Define your Weights and biases standard deviation for training
Wstd=input("Please define the weights standard deviation for training: ")
Bstd=input("Please define the bias standard deviation for training: ")

Wstd = float(Wstd)
Bstd = float(Bstd)

#### Do you want binary weights?
isBin = input("Do you want binary weights? yes or no: ")
if(isBin == 'yes'):
	isBin = 1
	isBin2 = 'yes'
else:
	isBin = 0
	isBin2 = 'no'	
#### Select network to train
N = input("Please select the network you want to train: \n 0. No reg \n 1. With Dropout \n 2. With DropConnect \n 3. With A-Connect \n Option: ")
N = int(N)
if(N==0):
	isNet = 0	#For no reg
elif N==1:
	isNet = 1  #For dropout
elif N==2:
	isNet = 2  #For dropconnect
elif N==3:
	isNet = 3  #For Aconnect		
	

if(N==0):
	string = "no_reg_network"
elif(N==1):
	string = "dropout_network"
elif(N==2):
	string = "dropconnect_network"
elif(N==3):
	string = "aconnect_network"
	if(isBin==1):
		if(Wstd !=0 or Bstd !=0):
			string = string+'_bw'
		else:
			string='FCquant_network'
	else:
		string = string
elif(N==4):
	string = "FCquant_network"	

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
	
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
model = MNIST_mismatch.Test_MNIST(N,imgsize,Wstd,Bstd,isBin2)


print('\n\n*******************************************************************************************\n\n')
print('TRAINING NETWORK: ', string)
print('\n\n*******************************************************************************************')
model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=batch_size)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
## Saving the training data
np.savetxt('./Models/Training data/'+string+'_acc'+'.txt',acc,fmt="%.2f")
np.savetxt('./Models/Training data/'+string+'_val_acc'+'.txt',val_acc,fmt="%.2f")

string = './Models/'+string+'.h5'
model.save(string,include_optimizer=True)

options = [str(Opt), str(Wstd), str(Bstd), str(isBin), str(isNet)]
data = "Train_Options.txt"
f = open(data,"w")
for i in options:
	f.write(i+'\n')
f.close()




