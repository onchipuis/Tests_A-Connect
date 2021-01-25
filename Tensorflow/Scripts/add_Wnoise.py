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
					if force == "no":
						Wstd = layers[i].Wstd
				if hasattr(layers[i], 'Bstd'):
					if force == "no":
						Bstd = layers[i].Wstd
				if hasattr(layers[i],'Werr'):
					Werr = abs(1+Wstd*Merr_aux)
					Berr = abs(1+Bstd*MBerr_aux)
					
					if hasattr(layers[i], 'Werr'):
						if(layers[i].Wstd != 0):
							layers[i].Werr[1,:,:] = Werr
						else:
							layers[i].Werr = np.random.randn()
					if hasattr(layers[i], 'Berr'):
						if(layers[i].Wstd != 0):
							layers[i].Berr[1,:,:] = Berr
						else:
							layers[i].Berr = np.random.randn()
				else:
					Werr = abs(1+Wstd*Merr_aux)
					layers[i].weights[0] = layers[i].weights[0] * Werr
					layers[i].W = layers[i].W * Werr
					Berr = abs(1+Bstd*MBerr_aux)
					layers[i].weights[1] = layers[i].weights[1] * Berr
					layers[i].bias = layers[i].bias * Berr
				
	return net,Wstd,Bstd
				
