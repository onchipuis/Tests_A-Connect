# A-Connect for TensorFlow

***

## General Info

All the work done to provide an A-Connect layer that works on Keras/Tensorflow is available here.

### Table of Contents

1. [Layers](/Tensorflow/Layers): All the layer created to test the algorithm. Incluiding the source code for an A-Connect layer. Please see local readme.
2. [Networks](/Tensorflow/Networks): All the networks used for testing purpose. Please see local readme.
3. [Useful Scripts](/Tensorflow/Scripts): Some useful scripts for training and training. Please see local readme.
4. [Graphs](/Tensorflow/Graphs): Some graphs that shows the algorithm performance and many other results.
5. [Trained models](/Tensorflow/Models): Finally, here you can find some trained networks to test. Please see local readme.

### Contents description and instructions.

Firstly run config.sh if you are using any Linux distribution. If you are not a Linux user, please run get_dir.py.

1. test_mnist.py: Script to train any neural network using the standard MNIST dataset. Also with this script you can use a custom training loop.
2. handwirtten.py: Provides a way to train a simple neural network using the standard MNIST dataset.
3. Train_Networks.py: Script to train 4 different neural networks. Without regularizations, with dropout after the first FC layer, with dropconnect on the first FC layer
and finally, with A-Connect. This scripts needs > 8GB RAM installed to run properly. Also could run in parallel using python Pool function.
4. MNIST_MCSim.py: This script performs a Monte Carlo simulation of the neural networks trained and saved in the layer Models.





