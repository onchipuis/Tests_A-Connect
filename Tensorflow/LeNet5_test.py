import tensorflow as tf
import numpy as np
from Networks import LeNet5
from Scripts import load_ds

(x_train, y_train), (x_test, y_test) = load_ds.load_ds()

model,x_train,x_test=LeNet5.LeNet5(x_train,x_test)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
model.compile(optimizer=optimizer,loss=['sparse_categorical_crossentropy'],metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train,y_train,validation_split=0.2,epochs = 20,batch_size=256)


