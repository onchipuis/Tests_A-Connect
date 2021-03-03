
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from Scripts import classify
from Scripts import add_Wnoise
import matplotlib.pyplot as plt
from Layers import fullyconnected
import tensorflow as tf

from Layers import AConnect
from Scripts import load_ds
from datetime import datetime
import matplotlib.pyplot as plt
config = open('config.txt','r')
folder = config.read()

def MCsim(net,Xtest,Ytest,M,Wstd,Bstd,force,Derr=0,net_name="Network",custom_objects=None,optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9),loss=['sparse_categorical_crossentropy'],metrics=['accuracy']):
	acc_noisy = np.zeros((M,1))
	#f = open(net_name,'w')
	local_net = tf.keras.models.load_model(net,custom_objects = custom_objects)
	local_net.save_weights(filepath=('./Models_New/'+net_name+'_weights.h5'))
	print(local_net.summary())
	print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy\n')
	print('----------------------------------------------------------------')
#	global parallel

	for i in range(M):
		[NetNoisy,Wstdn,Bstdn] = add_Wnoise.add_Wnoise(local_net,Wstd,Bstd,force,Derr)
		NetNoisy.compile(optimizer,loss,metrics)
		acc_noisy[i] = classify.classify(NetNoisy, Xtest, Ytest)
		acc_noisy[i] = 100*acc_noisy[i]
		print('\t%i\t | \t%.1f\t | \t%.1f\t | \t%.2f\n' %(i,Wstd*100,Bstd*100,acc_noisy[i]))
		local_net.load_weights(filepath=('./Models_New/'+net_name+'_weights.h5'))
#		return acc_noisy

	#pool = Pool(mp.cpu_count())
	#acc_noisy = pool.map(parallel, range(M))
	#pool.close()
	media = np.median(acc_noisy)
	print('----------------------------------------------------------------')
	print('Median: %.1f%%\n' % media)
	#print('IQR Accuracy: %.1f%%\n',100*iqr(acc_noisy))
#	print('Min. Accuracy: %.1f%%\n' % 100.0*np.amin(acc_noisy))
#	print('Max. Accuracy: %.1f%%\n'% 100.0*np.amax(acc_noisy))

	np.savetxt('./Results_New/'+net_name+'.txt',acc_noisy,fmt="%.2f")
	return acc_noisy, media
	
	
batch_size = 256
for Opt in range(4):

	if(Opt == 0): #For standard MNIST 28x28 8 bits
		imgsize = [28,28]
		Q = 8
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		
		f1 = open("Train_Options1.txt",'r')
		options1 = f1.readlines()
		#Opt = int(options[0])
		Wstd = float(options1[1])	
		Bstd = float(options1[2])	
		isBin = int(options1[3])
		isNet = int(options1[4])
		
		acc_noisy = np.zeros((1000,1))
		if(isNet == 3):
			string = "aconnect_network"
	
			if(isBin == 1):
				string = string+"_bw"
				N = 5
			elif(isBin == 0):
				N = 4
			else:
				print('F')
		
		
			string = string+"_28x28"
			string = string +"_8b"
			string = string+"_"+str(int(100*Wstd))+"_"+str(int(100*Bstd))+".h5"
			
			net = "./Models_New/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_28x28_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
			
		now = datetime.now()
		starttime = now.time()
		#####
		print('\n\n*******************************************************************************************\n\n')
		print('TESTING NETWORK: ', name)
		print('\n\n*******************************************************************************************')
		acc_noisy, media = MCsim(net,x_test,y_test,1000,0.5,0.5,"no",0,name,custom_objects)
		#####
		now = datetime.now()
		endtime = now.time()

		print('\n\n*******************************************************************************************')
		print('\n Simulation started at: ',starttime)
		print('Simulation finished at: ', endtime)
		
	elif(Opt==1): #For MNIST 28x28 4 bits
		imgsize = [28,28]
		Q = 4
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		
		f2 = open("Train_Options2.txt",'r')
		options2 = f2.readlines()
		#Opt = int(options[0])
		Wstd = float(options2[1])	
		Bstd = float(options2[2])	
		isBin = int(options2[3])
		isNet = int(options2[4])
		
		acc_noisy = np.zeros((1000,1))
		if(isNet == 3):
			string = "aconnect_network"
	
			if(isBin == 1):
				string = string+"_bw"
				N = 5
			elif(isBin == 0):
				N = 4
			else:
				print('F')
		
		
			string = string+"_28x28"
			string = string +"_4b"
			string = string+"_"+str(int(100*Wstd))+"_"+str(int(100*Bstd))+".h5"
			
			net = "./Models_New/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_28x28_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
	
		now = datetime.now()
		starttime = now.time()
		#####
		print('\n\n*******************************************************************************************\n\n')
		print('TESTING NETWORK: ', name)
		print('\n\n*******************************************************************************************')
		acc_noisy, media = MCsim(net,x_test,y_test,1000,0.5,0.5,"no",0,name,custom_objects)
		#####
		now = datetime.now()
		endtime = now.time()

		print('\n\n*******************************************************************************************')
		print('\n Simulation started at: ',starttime)
		print('Simulation finished at: ', endtime)
			
			
	elif(Opt==2): #For MNIST 11x11 8 bits
		imgsize = [11,11] 
		Q = 8
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		
		f3 = open("Train_Options3.txt",'r')
		options3 = f3.readlines()
		#Opt = int(options[0])
		Wstd = float(options3[1])	
		Bstd = float(options3[2])	
		isBin = int(options3[3])
		isNet = int(options3[4])
		
		acc_noisy = np.zeros((1000,1))
		if(isNet == 3):
			string = "aconnect_network"
	
			if(isBin == 1):
				string = string+"_bw"
				N = 5
			elif(isBin == 0):
				N = 4
			else:
				print('F')
		
		
			string = string+"_11x11"
			string = string +"_4b"
			string = string+"_"+str(int(100*Wstd))+"_"+str(int(100*Bstd))+".h5"
			
			net = "./Models_New/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_11x11_8b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
	
		now = datetime.now()
		starttime = now.time()
		#####
		print('\n\n*******************************************************************************************\n\n')
		print('TESTING NETWORK: ', name)
		print('\n\n*******************************************************************************************')
		acc_noisy, media = MCsim(net,x_test,y_test,1000,0.5,0.5,"no",0,name,custom_objects)
		#####
		now = datetime.now()
		endtime = now.time()

		print('\n\n*******************************************************************************************')
		print('\n Simulation started at: ',starttime)
		print('Simulation finished at: ', endtime)
		
		
	elif(Opt==3): #For MNIST 11x11 4 bits
		imgsize = [11,11] 
		Q = 4
		(x_train, y_train), (x_test, y_test) = load_ds.load_ds(imgsize,Q)
		
		f4 = open("Train_Options4.txt",'r')
		options4 = f4.readlines()
		#Opt = int(options[0])
		Wstd = float(options4[1])	
		Bstd = float(options4[2])	
		isBin = int(options4[3])
		isNet = int(options4[4])
		
		acc_noisy = np.zeros((1000,1))
		if(isNet == 3):
			string = "aconnect_network"
	
			if(isBin == 1):
				string = string+"_bw"
				N = 5
			elif(isBin == 0):
				N = 4
			else:
				print('F')
		
		
			string = string+"_11x11"
			string = string +"_4b"
			string = string+"_"+str(int(100*Wstd))+"_"+str(int(100*Bstd))+".h5"
			
			net = "./Models_New/"+string
			custom_objects = {'AConnect':AConnect.AConnect}
			name = "aconnect_bw_nn_11x11_4b"+'_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))
	
		now = datetime.now()
		starttime = now.time()
		#####
		print('\n\n*******************************************************************************************\n\n')
		print('TESTING NETWORK: ', name)
		print('\n\n*******************************************************************************************')
		acc_noisy, media = MCsim(net,x_test,y_test,1000,0.5,0.5,"no",0,name,custom_objects)
		#####
		now = datetime.now()
		endtime = now.time()

		print('\n\n*******************************************************************************************')
		print('\n Simulation started at: ',starttime)
		print('Simulation finished at: ', endtime)
		
		
	else:
		raise "Wrong option" 


	


	
		
		
		
