import tensorflow as tf
import numpy as np

def loss_model(model, x, y, training=True):
	y_ = model(x, training=training)
	return loss_object(y_true=y, y_pred=y_)
	
def gradients(model, inputs, targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model, inputs, targets)
	return loss_value, tape.gradient(loss_value, model.trainable_variables)
	
def optimizer(option="adam"):
	if option == "adam":
		return optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	elif option == "SGD":
		return optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
	else:
		print("Not supported optimizer")
		
def train_nn(model, x, y, epochs, batchSize=32, training=True, option="adam"):
	for i in range(epochs):
		epoch_loss_avg = tf.keras.metrics.Mean()
		epochs_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		
		for x,y 
