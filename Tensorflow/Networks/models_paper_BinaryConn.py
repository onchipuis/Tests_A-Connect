config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
import mylib as my
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
import tensorflow as tf
from Layers import AConnect
from Layers import ConvAConnect

def get_model(model,Wstd,Bstd,isBin,pool):
	if(model==0): #Fully Connected model for MNIST
		model = tf.keras.Sequential([
			tf.keras.layers.flatten(input_shape=[28,28]),
			AConnect.AConnect(34,Wstd,Bstd,isBin,pool),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('relu'),
			AConnect.AConnect(24,Wstd,Bstd,isBin,pool),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('relu'),
			AConnect.AConnect(3,Wstd,Bstd,isBin,pool),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('relu'),						
			AConnect.AConnect(10,Wstd,Bstd,isBin,pool),
			tf.keras.layers.Softmax()
		])	
	elif(model==1):
	
		model = tf.keras.Sequential([
			tf.keras.layers.InputLayer([32,32,3]),
			ConvAConnect.ConvAConnect(),
		])
