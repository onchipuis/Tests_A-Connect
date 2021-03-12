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
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(227,227),           
			tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
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

		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=[32,32,3]),
			tf.keras.layers.experimental.preprocessing.Resizing(227,227),    
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
	    ])


	return model
	
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']




model=model_creation(isAConnect=False)
#parametros para el entrenamiento
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9), metrics=['accuracy'])
print(model.summary())
model.fit(train_images,train_labels,
          batch_size=256,epochs=25,
          validation_split=0.2
          )

