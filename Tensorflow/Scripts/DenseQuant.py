import numpy as np
import tensorflow as tf
from FC_quant import FC_quant


class DenseQuant(tf.keras.Sequential):
	def train_step(self, data):
		x,y = data
		
		with tf.GradientTape(persistent=True) as tape:
		
			y_pred = self(x, training = True)
			loss = self.compiled_loss(y,y_pred, regularization_losses=self.losses)
	
		trainable_vars = self.trainable_variables
		weights = trainable_vars[1]
		print(weights)
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		

		# Update metrics
		self.compiled_metrics.update_state(y, y_pred)

		
		return {m.name: m.result() for m in self.metrics}

		
#def fcquant(outputSize):
#	layer = FC_quant(outputSize)
#	model = DenseQuant(layer)
#	return model


	
