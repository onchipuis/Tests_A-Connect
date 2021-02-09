import tensorflow as tf
import numpy as np

class ACModel(tf.keras.Sequential):

	def train_step(self, data):
	
		x, y = data
		
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True) # Do the forward computation
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
			
		gradients = tape.gradient(loss, self.trainable_variables) #Get the gradients

		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update the weights
#		tf.print(self.layers[3].W[1,:])
	#	tf.print(self.layers[1].bias[:])
		self.compiled_metrics.update_state(y, y_pred) #Update the metrics
		
		return {m.name: m.result() for m in self.metrics}
			
			
