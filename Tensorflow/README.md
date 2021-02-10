# A-Connect for TensorFlow

***

## General Info

All the work done to provide an A-Connect layer that works on Keras/Tensorflow is available here.

### Instructions and dependencies

All the work here it was done using Python 3.8 and Tensorflow 2.4.X, so please, in order to avoid some compatibility problems, try to work using the same version. (At least Python 3.X and Tensorflow 2.4.1). Also, you will need Python libraries like ***Numpy*** and ***Matplotlib*** and the ***Tensorflow Probability*** library. Finally, a instruction list is shown below.

1. Install Python 3.X.
2. Install Numpy.
3. Install Matplotlib.
4. [Install Tensorflow 2.4.1.] (https://www.tensorflow.org/install)
5. [Install Tensorflow Probability.] (https://www.tensorflow.org/probability/install)
6. Finally run config.sh (if you are a Linux user if not, please run get_dir.py).



### Table of Contents

1. [Layers](/Tensorflow/Layers): All the layer created to test the algorithm. Incluiding the source code for an A-Connect layer. Please see local readme.
2. [Networks](/Tensorflow/Networks): All the networks used for testing purpose. Please see local readme.
3. [Useful Scripts](/Tensorflow/Scripts): Some useful scripts for training and training. Please see local readme.
4. [Graphs](/Tensorflow/Graphs): Some graphs that shows the algorithm performance and many other results.
5. [Trained models](/Tensorflow/Models): Finally, here you can find some trained networks to test. Please see local readme.

### Contents description and instructions.

Firstly run config.sh if you are using any Linux distribution. If you are not a Linux user, please run get_dir.py.

1. [Test a neural network with MNIST 28X28](test_mnist.py): Script to train any neural network using the standard MNIST dataset. Also with this script you can use a custom training loop.
2. [Train in serial/parallel 4 NN](Train_Networks.py): Script to train 4 different neural networks. Without regularizations, with dropout after the first FC layer, with dropconnect on the first FC layer
and finally, with A-Connect. This script needs > 8GB RAM installed to run properly. Also could run in parallel using python Pool function.
3. [Monte Carlo simulation](MNIST_MCSim.py): This script performs a Monte Carlo simulation of the neural networks trained and saved in the layer Models.





