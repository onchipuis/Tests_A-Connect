import numpy as np
import tensorflow as tf
from Networks import MNIST_mismatch
import multiprocessing as mp
from multiprocessing import Pool
import time

from Scripts import MCsim
from Layers import DropConnect
from Layers import AConnect
from Layers import fullyconnected
from Layers import FC_quant
from Scripts import load_ds
from datetime import datetime
import matplotlib.pyplot as plt
config = open('config.txt','r')
folder = config.read()

f = open("Train_Options.txt",'r')
options = f.readlines()
Opt = int(options[0])
Wstd = float(options[1])	
Bstd = float(options[2])	
isBin = int(options[3])
isNet = int(options[4])

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
	raise "Wrong option"

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
		raise "Not supported matrix quantization"


acc_noisy = np.zeros((1000,1))
if(isNet == 0):
	string = "no_reg_network"
	N = 1
	if(Opt == 1 or Opt == 2):
		string = string+"_28x28"
		if(Opt == 1):
			string = string +"_8b"
		else:
			string = string +"_4b"
	else:
		string = string +"_11x11"
		if(Opt == 3):
			string = string +"_8b"
		else:
			string = string +"_4b"
	name = string			
	string = string+".h5"
	
if(isNet == 1):
	string = "dropout_network"
	N = 2
	if(Opt == 1 or Opt == 2):
		string = string+"_28x28"
		if(Opt == 1):
			string = string +"_8b"
		else:
			string = string +"_4b"
	else:
		string = string +"_11x11"
		if(Opt == 3):
			string = string +"_8b"
		else:
			string = string +"_4b"
	name = string			
	string = string+".h5"

if(isNet == 2):
	string = "dropconnect_network"
	N = 3
	if(Opt == 1 or Opt == 2):
		string = string+"_28x28"
		if(Opt == 1):
			string = string +"_8b"
		else:
			string = string +"_4b"
	else:
		string = string +"_11x11"
		if(Opt == 3):
			string = string +"_8b"
		else:
			string = string +"_4b"
	name = string			
	string = string+".h5"		

if(isNet == 3):
	string = "aconnect_network"
	
	if(isBin == 1):
		string = string+"_bw"
		N = 5
	elif(isBin == 0):
		N = 4
	else:
		print('F')
	if(Opt == 1 or Opt == 2):
		string = string+"_28x28"
		if(Opt == 1):
			string = string +"_8b"
		else:
			string = string +"_4b"
	else:
		string = string +"_11x11"
		if(Opt == 3):
			string = string +"_8b"
		else:
			string = string +"_4b"
	string = string+"_"+str(int(100*Wstd))+"_"+str(int(100*Bstd))+".h5"
		

if N == 1:
	net = "./Models/"+string
	custom_objects = None
	name = name
elif N == 2:
	net = "./Models/"+string
	custom_objects = None
	name = name
elif N == 3:
	net = "./Models/"+string
	custom_objects = {'DropConnect':DropConnect.DropConnect}
	name = name
elif N == 4:
	if(imgsize == [11,11]):
		if(Q==4):
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_nn_11x11_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
		else:
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_nn_11x11_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
	else:			
		if(Q==4):
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_nn_28x28_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
		else:
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_nn_28x28_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
			
elif N == 5:
	if(imgsize == [11,11]):
		if(Q==4):
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_11x11_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
		else:
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_11x11_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
	else:			
		if(Q==4):
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_28x28_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
		else:
			net = "./Models/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_28x28_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))			
"""		
elif N == 5:
	if(imgsize == [11,11]):
		if(Q==4):
			net = "./Models/FCquant_network_11x11_4b.h5"
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "FC_quant_nn_11x11_4b"
		else:
			net = "./Models/FCquant_network_11x11_8b.h5"
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "FC_quant_nn_11x11_8b"
	else:
		if(Q==4):
			net = "./Models/FCquant_network_28x28_4b.h5"
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "FC_quant_nn_28x28_4b"
		else:
			net = "./Models/FCquant_network_28x28_8b.h5"
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "FC_quant_nn_28x28_8b" """				
		






now = datetime.now()
starttime = now.time()
#####
print('\n\n*******************************************************************************************\n\n')
print('TESTING NETWORK: ', name)
print('\n\n*******************************************************************************************')
acc_noisy, media = MCsim.MCsim(net,x_test,y_test,1000,0.5,0.5,"no",0.9,name,custom_objects)
#####
now = datetime.now()
endtime = now.time()

print('\n\n*******************************************************************************************')
print('\n Simulation started at: ',starttime)
print('Simulation finished at: ', endtime)

#folder = folder+'/Graphs/'
#plt.title('Validation Accuracy Wstd = ' + Wstd)
#plt.xlabel('Median : %.2f' % media)
#plt.hist(acc_noisy)
#plt.grid(True)
#plt.savefig(folder+name+'.png')
#plt.show()
#plt.clf()


