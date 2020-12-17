import numpy as np


def add_Wnoise(net,Wstd,Bstd,force):
	layers = net.Layers # Buscar equivalente en tensorflow
	Nlayers = np.size(layers)
	
	SRAMsz = [1024 1024]
	SRAMBsz = [1024 1]
	
	Merr = np.random.randn(SRAMsz)
	MBerr = np.random.randn(SRAMBsz)
	
	for i in range(Nlayers):
		if hasattr(layers(i),'weights') or hasattr(layers(i),'W'): # Buscar equiv tf
			
			Wsz = np.shape(layers(i).Weights)
			Bsz = np.shape(layers(i).Bias)
			Merr_aux = Merr(1:Wsz(1), 1:Wsz(2))
			MBerr_aux = MBerr(1:Bsz(1), 1:Bsz(2))
			
			if hasattr(layers(i), 'Wstd'):
				if force == "no":
					Wstd = layers(i).Wstd
			if hasattr(layers(i), 'Bstd'):
				if force == "no":
					Bstd = layers(i).Bstd
			if hasattr(layers(i),'Werr'):
				Werr = abs(1+Wstd*Merr_aux)
				Berr = abs(1+Bstd*MBerr_aux)
				
				if hasattr(layers(i), 'Werr'):
					layers(i).Werr = Werr
				if hasattr(layers(i), 'Berr'):
					layers(i).Berr = Berr
			else:
				Werr = abs(1+Wstd*Merr_aux)
				layers(i).weights = layers(i).weights * Werr
				layers(i).W = layers(i).W * Werr
				Berr = abs(1+Bstd*MBerr_aux)
				layers(i).bias = layers(i).bias * Berr
				
	return net,Wstd,Bstd
				
