import numpy as np


def add_Wnoise(net,Wstd,Bstd,force):
	layers = net.Layers # Buscar equivalente en tensorflow
	Nlayers = np.size(layers)
	
	SRAMsz = [1024 1024]
	SRAMBsz = [1024 1]
	
	Merr = np.random.randn(SRAMsz)
	MBerr = np.random.randn(SRAMBsz)
	
	for i in range(Nlayers):
		if isprop(layers(i),'Weights'): # Buscar equiv tf
			
			Wsz = np.shape(layers(i).Weights)
			Bsz = np.shape(layers(i).Bias)
			Merr_aux = Merr(1:Wsz(1), 1:Wsz(2))
			MBerr_aux = MBerr(1:Bsz(1), 1:Bsz(2))
			
			if isprop(layers(i), 'Wstd'):
				if force == "no":
					Wstd = layers(i).Wstd
			if isprop(layers(i), 'Bstd'):
				if force == "no":
					Bstd = layers(i).Bstd
					
			if isprop(layers(i), 'Werr')
