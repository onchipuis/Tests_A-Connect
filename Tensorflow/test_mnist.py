import tensorflow as tf
import numpy as np
import Networks.mnist

#Option 1: Weights quantizied + custom backprop
#Option 2: Custom network dense layer+dropout
#Option 3: Only weights quantizied
#Option 4: Dropconnect network

start = "1"

while(start == "1"):
	option = int(input("Which network do you want to test: "))
	loss, acc = Networks.mnist.mnist_test(option)

	print("loss: %.4f" %float(loss))
	print("acc: %.4f"%float(100*acc))
	start = input("To continue training type 1 or any other word to exit: ")
