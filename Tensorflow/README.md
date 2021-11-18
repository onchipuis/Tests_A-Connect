# A-Connect for TensorFlow

***

## General Info

All the work done to provide an A-Connect layer that works on Keras/Tensorflow is available here. 

### Instructions and dependencies

All the work here it was done using Python 3.8 and Tensorflow 2.4.X, so please, in order to avoid some compatibility problems, try to work using the same version. (At least Python 3.X and Tensorflow 2.4.1). Also, you will need Python libraries like ***Numpy*** and ***Matplotlib*** and the ***Tensorflow Probability*** library. Finally, a instruction list is shown below.

1. Install Python 3.X.
2. Install Numpy.
3. Install Matplotlib.
4. [Install Tensorflow 2.4.X](https://www.tensorflow.org/install)
5. [Install Tensorflow Probability](https://www.tensorflow.org/probability/install)
6. Finally run config.sh (if you are a Linux user. If not, please run get_dir.py).


### Table of Contents
1. [aconnect](/Tensorflow/aconnect): This folder contains the final library for the methodology.
2. [Layers](/Tensorflow/Layers): All the layer created to test the algorithm. Incluiding the source code for an A-Connect layer. Please see local readme.
3. [Networks](/Tensorflow/Networks): All the networks used for testing purpose. Please see local readme.
4. [Useful Scripts](/Tensorflow/Scripts): Some useful scripts for training and training. Please see local readme.
5. [Graphs](/Tensorflow/Graphs): Some graphs that shows the algorithm performance and many other results.
6. [Trained models](/Tensorflow/Models): Here you can find some trained networks to test. Please see local readme.
7. [Results](/Tensorflow/Results): This folder contains the results of the Monte Carlo simulations.

### Content description

#### Scripts

1. [mylib](/Tensorflow/mylib.py): Library with some useful functions for visualizing the performance of the neural networks.
2. [NN_test](/Tensorflow/NN_test.py): Scripts for training and testing the performance of some networks from [MNIST_Mismatch](/Tensorflow/Networks/MNIST_mismatch.py).
3. [LeNet5_test](/Tensorflow/LeNet5_test.py): Script for training and testing the LeNet-5 architecture presented in [MNIST_Mismatch](/Tensorflow/Networks/LeNet5.py).
4. [get_dir](/Tensorflow/get_dir.py): Script for creating the file path for all the project to avoid some issues.






