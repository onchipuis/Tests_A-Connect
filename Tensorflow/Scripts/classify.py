### Edited by: Ricardo V. 11/12/2020
###
### Script to obtain the accuracy from predictions

import numpy as np

def classify(net,Xtest,Ytest):
	Ypred = net.predict(Xtest)
	accuracy = np.sum(Ypred == Yest)/np.size(Ytest)
