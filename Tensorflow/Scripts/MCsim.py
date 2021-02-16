### Edited by: Ricardo Vergel - 11/12/2020
###
### Monte Carlo simulation for testing neural networks


import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from Scripts import classify
from Scripts import add_Wnoise
import matplotlib.pyplot as plt
from Layers import fullyconnected
import tensorflow as tf

def MCsim(net,Xtest,Ytest,M,Wstd,Bstd,force,Derr=0,net_name="Network",custom_objects=None,optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),loss=['sparse_categorical_crossentropy'],metrics=['accuracy']):
	acc_noisy = np.zeros((M,1))
	#f = open(net_name,'w')
	local_net = tf.keras.models.load_model(net,custom_objects = custom_objects)
	local_net.save_weights(filepath=('./Models/'+net_name+'_weights.h5'))
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
		local_net.load_weights(filepath=('./Models/'+net_name+'_weights.h5'))
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

	np.savetxt('../Results'+net_name+'.txt',acc_noisy,fmt="%.2f")
	return acc_noisy, media
	
