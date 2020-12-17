### Edited by: Ricardo Vergel - 11/12/2020
###
### Script to do a Monte Carlo simulation to test neural networks


import numpy as np
from multiprocessing import Pool
from classify import classify


def MCsim(net,Xtest,Ytest,M,Wstd,Bstd,force):
	acc_noisy = np.zeros(M,1)
	print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy\n')
	print('----------------------------------------------------------------')
	pool = Pool()
	chunkSize = 20
	
	def parallel(i):
		[NetNoisy,Wstdn,Bstdn] = add_Wnoise(net,Wstd,Bstd,force)
		acc_noisy(i) = classify(NetNoisy, Xtest, Ytest)
		print('\t%i\t | \t%f\t | \t%f\t | \t%f\n' %(i,Wstdn*100,Bstdn*100,100*acc_noisy(i)))
		return acc_noisy
	acc_noisy = pool.map(parallel, range(M))
	print('----------------------------------------------------------------')
	print('Median Accuracy: %.1f%%\n' % 100*np.median(acc_noisy))
	#print('IQR Accuracy: %.1f%%\n',100*iqr(acc_noisy))
	print('Min. Accuracy: %.1f%%\n' % 100*np.amin(acc_noisy))
	print('Max. Accuracy: %.1f%%\n'% 100*np.amax(acc_noisy))

	
