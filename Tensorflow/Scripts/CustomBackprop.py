import tensorflow as tf
import numpy as np



class CustomBackprop(tf.keras.Sequential):
	def train_step(self, data):
		x, y = data
		#Wrandom = tf.random.normal(shape=(x.shape[0],x.shape[1]),mean=10,stddev=0.7)
		with tf.GradientTape() as tape:
			y_pred = self(x, training = True)
			loss = self.compiled_loss(y,y_pred, regularization_losses=self.losses)
	
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		

		# Update metrics
		self.compiled_metrics.update_state(y, y_pred)

		
		return {m.name: m.result() for m in self.metrics}


	
