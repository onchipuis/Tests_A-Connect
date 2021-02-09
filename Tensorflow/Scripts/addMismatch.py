import numpy as np
import tensorflow as tf

def addMismatch(layer):
	batchSize = layer.batch_size
	if(layer.Wstd != 0):
		ID = range(np.size(layer.Werr,0))
		ID = tf.random.shuffle(ID)

	if(layer.isBin == "yes"):
		weights = tf.math.sign(layer.W)
	else:
		weights = layer.W
		
	Werrdb_sz = tf.shape(layer.Werr)
	
	if(tf.size(Werrdb_sz) == 3):
		loc_id = tf.slice(ID, [0], [batchSize])
		Werr = tf.gather(layer.Werr,[loc_id])
		Werr = tf.squeeze(Werr)
	#elif(np.size(Werrdb_sz) == 2):
	#	Werr = tf.tile(layer.Werr,batchSize)
	else:
		Werr = tf.constant(1,dtype=tf.float32)

	weights = tf.math.multiply(Werr,weights)
#	tf.print(weights)

	if(layer.isBin == "yes"):
		weights = weights/layer.W
		
	Berrdb_sz = tf.shape(layer.Berr)
	bias = layer.bias
																																																			
	if(tf.size(Berrdb_sz) == 3):
		Berr = tf.gather(layer.Berr, [loc_id])
		Berr = tf.squeeze(Berr,axis=0)
	#elif(np.size(Berrdb_sz) == 2):
	#	Berr = tf.tile(layer.Berr,batchSize)
	else:
		Berr = tf.constant(1,dtype=tf.float32)
		
	bias = tf.math.multiply(Berr,bias)
	
	return [weights, bias]


