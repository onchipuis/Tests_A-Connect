import numpy as np
import tensorflow as tf

def addMismatch(layer, batchSize):
	if(np.size(layer.Werr)!=1):
		dim = np.shape(layer.Werr)
		ID = np.random.randint(0,dim[0]-1)
	else:
		ID = 1

	if(layer.isBin == "yes"):
		weights = tf.math.sign(layer.W)
	else:
		weights = layer.W
		
	Werrdb_sz = np.shape(layer.Werr)
	
	if(np.size(Werrdb_sz) == 3):
		Werr = layer.Werr[ID,:,:]
	elif(np.size(Werrdb_sz) == 2):
		Werr = np.title(layer.Werr,1,1,batchSize)
	else:
		Werr = 1
		
	weights = Werr*weights
	
	
	if(layer.isBin == "yes"):
		weights = weights/layer.W
		
	Berrdb_sz = np.size(layer.Berr)
	bias = layer.bias
	
	if(np.size(Berrdb_sz) == 3):
		Berr= layer.Berr[ID,:,:]
	elif(np.size(Berrdb_sz) == 2):
		Berr = np.title(layer.Berr,1,1,batchSize)
	else:
		Berr = 1
		
	bias = Berr*bias
	
	return [weights, bias, Werr, Berr]


