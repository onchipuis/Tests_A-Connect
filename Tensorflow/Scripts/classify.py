### Edited by: Ricardo V. 11/12/2020
###
### Script to obtain the accuracy from predictions

import numpy as np

def classify(net,Xtest,Ytest):
	#pred = np.zeros(len(Ytest))
	#label_predicted = np.zeros(len(Ytest))
	#Ypred = net.predict(Xtest)

	#for i in range(len(Xtest)):
	#	temp = np.argmax(Ypred[i])
	#	label_predicted[i] = int(temp)
		#print(label_predicted[i])
	#	pred[i] = label_predicted[i] == Ytest[i]
		
	#accuracy = np.sum(pred)/len(Ytest)
	_,accuracy = net.evaluate(Xtest,Ytest,verbose=0,use_multiprocessing=True)
	return accuracy
