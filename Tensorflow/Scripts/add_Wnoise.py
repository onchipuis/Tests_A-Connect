import numpy as np


def add_Wnoise(net,Wstd,Bstd,force):
	layers = net.layers 
	Nlayers = np.size(layers)
	
	SRAMsz = [1024,1024]
	SRAMBsz = [1024,1]
	
	Merr = np.random.randn(SRAMsz[0],SRAMsz[1])
	MBerr = np.random.randn(SRAMBsz[0],SRAMBsz[1])
	
	for i in range(Nlayers):
		if layers[i].count_params() != 0:
			if hasattr(layers[i],'weights') or hasattr(layers[i],'W')or hasattr(layers[i],'w'): 
				
				Wsz = np.shape(layers[i].weights[0])
				Bsz = np.shape(layers[i].weights[1])
				#print(Bsz[0])
				Merr_aux = Merr[0:Wsz[0], 0:Wsz[1]]
				MBerr_aux = MBerr[1,0:Bsz[0]]
				
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
				if hasattr(layers[i],'Werr'):
					
					Werr = abs(1+Wstd*Merr_aux)
					Berr = abs(1+Bstd*MBerr_aux)
					
					if hasattr(layers[i], 'Werr'):
						if(layers[i].Wstd != 0):
							if(np.size(np.shape(layers[i].Werr)) == 3):
								layers[i].Werr[1,:,:] = Werr
							else:
								layers[i].Werr = Werr
						else:
							layers[i].Werr = np.random.randn()
					if hasattr(layers[i], 'Berr'):
						if(layers[i].Wstd != 0):
							if(np.size(np.shape(layers[i].Berr)) == 3):
								layers[i].Berr[1,:,:] = Berr
							else:
								layers[i].Berr = Berr
						else:
							layers[i].Berr = np.random.randn()
				else:
					Werr = abs(1+Wstd*Merr_aux)
					Berr = abs(1+Bstd*MBerr_aux)
					local_weights = [layers[i].weights[0]*Werr,layers[i].weights[1]*Berr]
					layers[i].set_weights(local_weights)
					#layers[i].weights[1] = layers[i].set_weights(layers[i].weights[1]*Berr)


					
				
					
				
	return net,Wstd,Bstd
				
