import tensorflow as tf
import numpy as np


x_train, y_train, x_test, y_test = tf.keras.datasets.cifar10.load_data()

def process_images(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
	#image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
	image = tf.image.resize(image, (224,224))
	return image	
	
x_train	= process_images(x_train)
x_test = process_images(x_test)

train_ds = tf.data.Dataset.from_tensor_slices(x_train,y_train)
test_ds = tf.data.Dataset.from_tensor_slices(x_test,y_test)

path = '../Images/Cifar10_resized'

tf.data.experimental.path(train_ds,path+'/Training')
tf.data.experimental.path(test_ds,path+'/Testing')
