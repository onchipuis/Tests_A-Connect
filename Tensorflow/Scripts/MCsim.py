### Edited by: Ricardo Vergel - 11/12/2020
###
### Monte Carlo simulation for testing neural networks
### How to
"""
Input Parameters:
net: Name of the network model you want to test (it must be saved in the folder Models)
Xtest and Ytest: Validation/Testing dataset
M: Number of samples for the Monte Carlo
Wstd and Bstd: Weights and Bias error for the simulation. It must be a float between 0-1
force: String, should be "yes" or "no" when you want to use a Wstd or Bstd different from the used during training i.e. If you trained A-Connect with 50% and you want to test it with an error of 70%
you must define force="yes"
Derr: If you want to introduce a deterministic error when you are using BW in the network. Float between 0-1
net_name = String with the name you want to use to save the simulation results
custom_objects: Python dictionary with the name of all the custom elements that you used in your model i.e. If you use an A-Connect model with Conv and FC A-Connect custom_objects should be
custom_objects= {'ConvAConnect':ConvAConnect.ConvAConnect,'AConnect':AConnect.AConnect}
SRAMsz: Matrix dimension for the static error matrix that you want to generate. It is depend on the dimension of the layer weights
SRAMBsz: Vector dimension for the static error vector that you want to generate. It is depend on the dimension of the layer weights
optimizer,loss,metrics: The values that you used during the training
This function returns the noisy accuracy values and the mean of this values
"""
import numpy as np
from multiprocessing import Pool
from Scripts import classify
from Scripts import add_Wnoise
import matplotlib.pyplot as plt
from Layers import fullyconnected
import tensorflow as tf

def MCsim(net,Xtest,Ytest,M,Wstd,Bstd,force,Derr=0,net_name="Network",custom_objects=None,dtype='float32',
optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),loss=['sparse_categorical_crossentropy'],metrics=['accuracy']):
	acc_noisy = np.zeros((M,1)) #Initilize the variable where im going to save the noisy accuracy
	local_net = tf.keras.models.load_model(net,custom_objects = custom_objects) #Load the trained model
	local_net.save_weights(filepath=('./Models/'+net_name+'_weights.h5')) #Save the weights. It is used to optimize the script RAM consumption
	print(local_net.summary()) #Print the network summary
	print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy\n')
	print('----------------------------------------------------------------')
#	global parallel

	for i in range(M): #Iterate over M samples
		[NetNoisy,Wstdn,Bstdn] = add_Wnoise.add_Wnoise(local_net,Wstd,Bstd,force,Derr,dtype=dtype) #Function that adds the new noisy matrices to the layers
		NetNoisy.compile(optimizer,loss,metrics) #Compile the model. It is necessary to use the model.evaluate
		acc_noisy[i] = classify.classify(NetNoisy, Xtest, Ytest) #Get the accuracy of the network
		acc_noisy[i] = 100*acc_noisy[i]
		print('\t%i\t | \t%.1f\t | \t%.1f\t | \t%.2f\n' %(i,Wstd*100,Bstd*100,acc_noisy[i]))
		local_net.load_weights(filepath=('./Models/'+net_name+'_weights.h5')) #Takes the original weights value.
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

	np.savetxt('./Results/'+net_name+'_simerr_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))+'.txt',acc_noisy,fmt="%.2f") #Save the accuracy of M samples in a txt
	return acc_noisy, media
	
