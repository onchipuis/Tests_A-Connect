import tensorflow as tf
import numpy as np
import Networks.mnist
from Networks import MNIST_mismatch

#Option 1: Weights quantizied + custom backprop
#Option 2: Custom network dense layer+dropout
#Option 3: Only weights quantizied
#Option 4: Dropconnect network



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model, fname_test, fname_train = MNIST_mismatch.Test_MNIST(3)
model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test,),epochs = 10)
print(model.layers[1].ID)
