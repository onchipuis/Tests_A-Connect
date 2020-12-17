# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Custom layer
from fullyconnected import fullyconnected
#from FCA import FCA
#from DropLayer import DropLayer
from FC_quant import FC_quant
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import mylib as my
from CustomBackprop import CustomBackprop

def handwrite(custom="no"):
	handwritten_mnist = keras.datasets.mnist

	(train_images, train_labels), (test_images, test_labels) = handwritten_mnist.load_data()

	numbers = ['0','1','2','3','4','5','6','7','8','9']

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	if(custom == "yes"):
		layer = FC_quant(10)
	else:
		layer = keras.layers.Dense(10)

	model = keras.Sequential([

		keras.layers.InputLayer(input_shape=(28,28)),
		keras.layers.Reshape((28,28,1)),
		keras.layers.Conv2D(8, kernel_size= (3,3), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),

		keras.layers.Conv2D(16, kernel_size= (3,3), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),
		keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),

		keras.layers.Conv2D(32, kernel_size= (3,3), padding='same'),
		keras.layers.BatchNormalization(),
		keras.layers.ReLU(),

		keras.layers.Flatten(),
		keras.layers.Dropout(0.5),
		layer,
		keras.layers.Softmax()
		
	])
	
	if(custom == "yes"):
		m = CustomBackprop(model)
		model.compile(optimizer='adam',
		          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		          metrics=['accuracy'])
		history = model.fit(train_images, train_labels,validation_data=(test_images,test_labels), epochs=4)
	else:
		model.compile(optimizer='adam',
		          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		          metrics=['accuracy'])
		history = model.fit(train_images, train_labels,validation_data=(test_images,test_labels), epochs=4)

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)

	predictions = model.predict(test_images)
	my.plot_test_imgs(predictions,test_labels,test_images, numbers, 5 , 3)
	my.plot_full_history(history.history['accuracy'],history.history['val_accuracy'],history.history['loss'],history.history['val_loss'],range(4))

	return test_acc, m
