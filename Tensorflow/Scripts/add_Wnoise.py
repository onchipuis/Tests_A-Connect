import numpy as np
import tensorflow as tf

def add_Wnoise(net,Wstd,Bstd,force,optimizer='Adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy']):
	layers = net.layers 
	Nlayers = np.size(layers)
	
	SRAMsz = [1024,1024]
	SRAMBsz = [1024]
	
	Merr = np.random.randn(SRAMsz[0],SRAMsz[1])
	MBerr = np.random.randn(SRAMBsz[0])

#	
	for i in range(Nlayers):
		if layers[i].count_params() != 0:

			if hasattr(layers[i],'kernel') or hasattr(layers[i],'W'): 

				Wsz = np.shape(layers[i].weights[0])
				Bsz = np.shape(layers[i].weights[1])
				Merr_aux = Merr[0:Wsz[0], 0:Wsz[1]]
				MBerr_aux = MBerr[0:Bsz[0]]
				
				if hasattr(layers[i], 'Wstd'):
					if(layers[i].Wstd != 0):
						if force == "no":
							Wstd = layers[i].Wstd
					else:
						Wstd = Wstd
				else:
					Wstd = Wstd
				if hasattr(layers[i], 'Bstd'):
					if(layers[i].Bstd != 0):
						if force == "no":
							Bstd = layers[i].Bstd
					else:
						Bstd = Bstd
				else:
					Bstd = Bstd
				if hasattr(layers[i],'Werr') or hasattr(layers[i],'Berr'):
#					
					Werr = abs(1+Wstd*Merr_aux)
					Berr = abs(1+Bstd*MBerr_aux)
					if hasattr(layers[i], 'Wstd'):
						if(layers[i].Wstd != 0):
							layers[i].Werr[1,:,:] = Werr
						else:			
							layers[i].Werr = Werr
					else:
							layers[i].Werr = Werr					
					if hasattr(layers[i], 'Bstd'):
						if(layers[i].Bstd != 0):
							layers[i].Berr[1,:] = Berr
						else:
							layers[i].Berr = Berr
					else:
						layers[i].Berr = Berr
				else:

					Werr = abs(1+Wstd*Merr_aux)
					Berr = abs(1+Bstd*MBerr_aux)
					weights = layers[i].weights[0]*Werr
					bias = layers[i].weights[1]*Berr
					local_weights = [weights,bias]
					layers[i].set_weights(local_weights)

	

		
	#layers = [tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(10),tf.keras.layers.Softmax()]		
	NoisyNet = tf.keras.Sequential(layers)
	#net.compile(optimizer,loss,metrics)	
	return NoisyNet,Wstd,Bstd
				
