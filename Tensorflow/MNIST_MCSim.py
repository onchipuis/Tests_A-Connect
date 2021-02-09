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
from datetime import datetime
import matplotlib.pyplot as plt
config = open('config.txt','r')
folder = config.read()

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
#for i in range(4):
#	if i == 0:
#		model[i] = tf.keras.models.load_model("./Models/no_reg_network.h5",custom_objects = {'fullyconnected':fullyconnected.fullyconnected})
#	elif i == 1:
#		model[i] = tf.keras.models.load_model("./Models/dropout_network.h5")
#	elif i == 2:
#		model[i] = tf.keras.models.load_model("./Models/dropconnect_network.h5", custom_objects = {'DropConnect':DropConnect.DropConnect})
#	elif i == 3:
#		model[i] = tf.keras.models.load_model("./Models/aconnect_network.h5", custom_objects = {'AConnect':AConnect.AConnect})

acc_noisy = np.zeros((1000,1))
N = 4
Wstd = '50%'
if N == 1:
	net = "./Models/no_reg_network.h5"
	custom_objects = None
	name = "noreg_nn"
elif N == 2:
	net = "./Models/dropout_network.h5"
	custom_objects = None
	name = "dropout_nn"
elif N == 3:
	net = "./Models/dropconnect_network.h5"
	custom_objects = {'DropConnect':DropConnect.DropConnect}
	name = "dropconnect_nn"
elif N == 4:
	net = "./Models/aconnect_network.h5"
	custom_objects = {'AConnect':AConnect.AConnect}
	name = "aconnect_nn"

now = datetime.now()
starttime = now.time()
#####
acc_noisy, media = MCsim.MCsim(net,x_test,y_test,1000,0.5,0.5,"no",name,custom_objects)
#####
now = datetime.now()
endtime = now.time()

print('\n\n*******************************************************************************************')
print('\n Simulation started at: ',starttime)
print('Simulation finished at: ', endtime)

folder = folder+'/Graphs/'
plt.title('Validation Accuracy Wstd = ' + Wstd)
plt.xlabel('Median : %.2f' % media)
plt.hist(acc_noisy)
plt.grid(True)
plt.savefig(folder+name+'.png')
#plt.show()
plt.clf()


