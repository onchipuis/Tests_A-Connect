### Edited by: Ricardo Vergel - 11/12/2020
###
### Script to do a Monte Carlo simulation to test neural networks


import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from Scripts import classify
from Scripts import add_Wnoise
import matplotlib.pyplot as plt

def MCsim(net,Xtest,Ytest,M,Wstd,Bstd,force):
	acc_noisy = np.zeros((M,1))
	print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy\n')
	print('----------------------------------------------------------------')
	
#	global parallel
	for i in range(M):
		[NetNoisy,Wstdn,Bstdn] = add_Wnoise.add_Wnoise(net,Wstd,Bstd,force)
		acc_noisy[i] = classify.classify(NetNoisy, Xtest, Ytest)
		acc_noisy[i] = 100*acc_noisy[i]
		print('\t%i\t | \t%f\t | \t%f\t | \t%f\n' %(i,Wstd*100,Bstd*100,acc_noisy[i]))
#		return acc_noisy
	#pool = Pool(mp.cpu_count())
	#acc_noisy = pool.map(parallel, range(M))
	#pool.close()
	print('----------------------------------------------------------------')
#	print('Median Accuracy: %.1f%%\n' % 100.0*np.median(acc_noisy))
	#print('IQR Accuracy: %.1f%%\n',100*iqr(acc_noisy))
#	print('Min. Accuracy: %.1f%%\n' % 100.0*np.amin(acc_noisy))
#	print('Max. Accuracy: %.1f%%\n'% 100.0*np.amax(acc_noisy))

	plt.title('Accuracy')
	plt.hist(acc_noisy, range=((0,100)))
	plt.grid(True)
	plt.show()
	plt.clf()
	
