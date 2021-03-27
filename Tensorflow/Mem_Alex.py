import sys
config = open('config.txt','r')
folder = config.read()
sys.path.append(folder)
sys.path.append(folder+'/Layers/')
sys.path.append(folder+'/Scripts/')
from Layers import ConvAConnect
from Layers import AConnect
from Scripts import MCsim
import numpy as np
import tensorflow as tf
from memory_profiler import profile

@profile
def memory_test(Wstd,Bstd):
    from tensorflow.compat.v1.keras.backend import set_session
    from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions
    tf_config = ConfigProto(gpu_options=GPUOptions(allow_growth=False))
    session = Session(config=tf_config)
    set_session(session)
    model = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=[32,32,3]),
		tf.keras.layers.experimental.preprocessing.Resizing(227,227),    
		ConvAConnect.ConvAConnect(filters=96, kernel_size=(11,11),Wstd=Wstd,Bstd=Bstd, strides=4,padding="VALID",Op=2,Slice=1,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		ConvAConnect.ConvAConnect(filters=256, kernel_size=(5,5), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		ConvAConnect.ConvAConnect(filters=384, kernel_size=(3,3), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.BatchNormalization(),
		ConvAConnect.ConvAConnect(filters=384, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.BatchNormalization(),
		ConvAConnect.ConvAConnect(filters=256, kernel_size=(1,1), Wstd=Wstd,Bstd=Bstd, strides=1,padding="SAME",Op=2,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
		tf.keras.layers.Flatten(),
		AConnect.AConnect(512, Wstd=Wstd,Bstd=Bstd,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.Dropout(0.5),
		AConnect.AConnect(512, Wstd=Wstd,Bstd=Bstd,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.ReLU(),
		tf.keras.layers.Dropout(0.5),
		AConnect.AConnect(10, Wstd=Wstd,Bstd=Bstd,Slice=8,d_type=tf.dtypes.float16),
        tf.keras.layers.Softmax()
	    ])
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()	
    #CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, 
                                   horizontal_flip=True, vertical_flip=True)
    train_datagen.fit(train_images)
    train_data = train_datagen.flow(train_images,train_labels, batch_size = 256)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen.fit(test_images)
    val_data = val_datagen.flow(test_images, test_labels, batch_size = 256) 
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001,momentum=0.9), metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_data,batch_size=256,epochs=50,validation_data=val_data)    
memory_test(Wstd=0.3,Bstd=0.3)
