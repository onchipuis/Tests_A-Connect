import numpy as np
import tensorflow as tf
from Networks import MNIST_mismatch
import multiprocessing as mp
from multiprocessing import Pool
import time
from Scripts import MCsim
from Layers import DropConnect
from Layers import AConnect
from Scripts import classify
from Layers import fullyconnected
import matplotlib.pyplot as plt
import get_dir as gd
folder = gd.get_dir()

fname_test = ['','','','']
fname_train = ['','','','']
numbers = ['0','1','2','3','4','5','6','7','8','9']
predictions = [[],[],[],[]]


def load_ds():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train/255.0
	x_test = x_test/255.0
	return (x_train, y_train), (x_test, y_test) 
	

model = [[],[],[],[]]
history = [[],[],[],[]]
(x_train, y_train), (x_test, y_test) = load_ds()
for i in range(4):
	if i == 0:
		model[i] = tf.keras.models.load_model("./Models/no_reg_network.h5",custom_objects = {'fullyconnected':fullyconnected.fullyconnected})
	elif i == 1:
		model[i] = tf.keras.models.load_model("./Models/dropout_network.h5")
	elif i == 2:
		model[i] = tf.keras.models.load_model("./Models/dropconnect_network.h5", custom_objects = {'DropConnect':DropConnect.DropConnect})
	elif i == 3:
		model[i] = tf.keras.models.load_model("./Models/aconnect_network.h5", custom_objects = {'AConnect':AConnect.AConnect})

acc_noisy = np.zeros((1000,1))
net = "./Models/no_reg_network.h5"
custom_objects = {'fullyconnected':fullyconnected.fullyconnected}
acc_noisy = MCsim.MCsim(net,x_test,y_test,1000,0.5,0.5,"no",custom_objects)


folder = folder+'/Graphs/'
plt.title('Accuracy')
plt.hist(acc_noisy)
plt.grid(True)
plt.savefig(folder+'no_reg_nn'+'.png')
#plt.show()
plt.clf()


