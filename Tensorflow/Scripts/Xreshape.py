import numpy as np
import tensorflow as tf
def Xreshape(Xin):
	Xsz = np.shape(Xin)
	dim = np.size(Xsz)
	if(dim<=3):
		batchSize = 1
	else:
		batchSize = Xsz[-1]
		Xsz = Xsz[0:[-1]-1]
		
	Xrow = Xsz[-1]
	
	return batchSize, Xsz
