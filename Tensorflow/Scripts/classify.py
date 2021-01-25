### Edited by: Ricardo V. 11/12/2020
###
### Script to obtain the accuracy from predictions

import numpy as np

def classify(net,Xtest,Ytest):
	Ypred = net.predict(Xtest)
	pred = np.zeros(len(Ytest))
	for i in range(len(Ytest)):
		pred[i] = np.argmax(Ypred[i])
		pred[i] = int(pred[i])
		#print(pred[i])
	accuracy = np.sum(pred == Ytest)/np.size(Ytest)
	#_,accuracy = net.evaluate(Xtest,Ytest,verbose=2)
	return accuracy
