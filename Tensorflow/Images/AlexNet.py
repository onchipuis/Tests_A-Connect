import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')

import tensorflow as tf

def model_creation(isAConnect=False,Wstd=0,Bstd=0):
	if(not(isAConnect)):
		model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(4096, activation='relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(4096, activation='relu'),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
	    ])
	else:
		print("1")
		"""

		model = tf.keras.models.Sequential([
            tf.keras.InputLayer(input_shape=[224,224]),
		    ConvAConnect.ConvAConnect(filters=96, kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd, strides=4,padding="VALID"),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(5,5), Wstd=Wstd,Bstd=Bstd, strides=1,  padding="same"),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    ConvAConnect.ConvAConnect(filters=384, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, strides=1,  padding="same"),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=384, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,  padding="same"),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    ConvAConnect.ConvAConnect(filters=256, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,  padding="same"),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.BatchNormalization(),
		    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		    tf.keras.layers.Flatten(),
		    AConnect.AConnect(4096, Wstd=Wstd,Bstd=Bstd),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.Dropout(0.5),
		    AConnect.AConnect(4096 , Wstd=Wstd,Bstd=Bstd),
            tf.keras.layers.ReLU(),
		    tf.keras.layers.Dropout(0.5),
		    AConnect.AConnect(10, Wstd=Wstd,Bstd=Bstd),
            tf.keras.layers.Softmax()
	    ])"""


	return model
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
	image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
	image = tf.image.resize(image, (224,224))
	return image, label	
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
#CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#validation_images, validation_labels = train_images[:5000], train_labels[:5000]
#train_images, train_labels = train_images[5000:], train_labels[5000:]
#train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
#test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
#validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
#train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
#test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
#validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
#print("Training data size:", train_ds_size)
#print("Test data size:", test_ds_size)
#print("Validation data size:", validation_ds_size)
#train_ds = (train_ds
#                 .map(process_images)
#                 .shuffle(buffer_size=train_ds_size)
#                  .batch(batch_size=32, drop_remainder=True))
#test_ds = (test_ds"""
"""

                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))"""



model=model_creation(isAConnect=False)
train_images,train_labels=process_images(train_images,train_labels)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.1,momentum=0.9), metrics=['accuracy'])
print(model.summary())
model.fit(train_images,train_labels,
          batch_size=32,epochs=10,
          )

