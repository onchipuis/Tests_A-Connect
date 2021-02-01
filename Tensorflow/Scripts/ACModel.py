import tensorflow as tf
import numpy as np

class ACModel(tf.keras.Sequential):

	def train_step(self, data):
	
		x, y = data
	
		for i in range(len(self.layers)):
			if(hasattr(self.layers[i], 'Wstd')):
				if(self.layers[i].Wstd != 0):
					with tf.GradientTape() as tape:
						y_pred = self(x, training=True) # Do the forward computation
						loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
					gradients = tape.gradient(loss , [self.layers[i].memweights, self.layers[i].membias])
					#i = len(self.layers)
					gradients[0] = tf.math.reduce_sum(gradients[0], axis=0)
					gradients[1] = tf.math.reduce_sum(gradients[1], axis=0)
					self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update the weights
				else:
					with tf.GradientTape() as tape:
						y_pred = self(x, training=True) # Do the forward computation
						loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
					gradients = tape.gradient(loss, self.trainable_variables)
					self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update the weights
			else:
				with tf.GradientTape() as tape:
					y_pred = self(x, training=True) # Do the forward computation
					loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
				gradients = tape.gradient(loss, self.trainable_variables)
					
				self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update the weights
			
			self.compiled_metrics.update_state(y, y_pred) #Update the metrics
		
		return {m.name: m.result() for m in self.metrics}
			
			
