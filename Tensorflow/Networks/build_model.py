"""
Script to load some neural networks models:


HOW TO:

string: Select which neural network you want to train/test: "2FC": For a simple one-hidden layer network presented in the paper.
							    "LeNet": For the standard LeNet-5 using MNIST 28x28 and A-Connect.
							    "VGG": For a modified version of VGG-16 with CIFAR-10 upsampled to 100x100
imgsize: Size vector for MNIST and 2FC test. Supported values are [28,28] and [11,11]
Quant: Matrix quantization for MNIST dataset. Supported values are 8 and 4
Wstd: Weights standard deviation
Bstd: Bias standard deviation
isBin: String for weights binarization
pool: Number of error matrices to be used during the training process. Do not use the batch size as number if you want a fast training
"""

from LeNet5 import LeNet5
from 2FC_model import build_model
from VGG import model_creation
from aconnect.scripts import load_ds
from tensorflow.keras.datasets import cifar10

def build_model(string="2FC",imgsize=[28,28],Quant=8,Wstd=0,Bstd=0,isBin="no",pool=None):

	if string is "2FC" or "LeNet":
		if string is "2FC":
			model = build_model(imgsize,Wstd,Bstd,isBin,pool)
			(x_train, y_train),(x_test, y_test) = load_ds(imgSize=imgsize, Quant=Quant)
		else:
			(x_train, y_train),(x_test, y_test) = load_ds(imgSize=[28,28], Quant=Quant)
			model = LeNet5(Xtrain=x_train,Xtest=x_test,isAConnect=True,Wstd=Wstd,Bstd=Bstd,isBin=isBin,pool=pool)
	elif string is "VGG":
		model = model_creation(isAConnect=True,Wstd=Wstd,Bstd=Bstd)
		(x_train, y_train),(x_test, y_test) = cifar10.load_data()
	else:
		print("wrong option")

	Train_ds = (x_train, y_train)
	Test_ds = (x_test, y_test)

return model, Train_ds, Test_ds


