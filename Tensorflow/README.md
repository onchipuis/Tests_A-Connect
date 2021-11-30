# Testing A-Connect on Multiple Neural Networks (TensorFlow)

***

## General Info

All the work done to provide an A-Connect layer that works on Keras/Tensorflow is available here. 

### Instructions and dependencies

All the work here was done using Python 3.8 and Tensorflow 2.4.X, so please, in order to avoid some compatibility problems, try to work using the same version. (At least Python 3.X and Tensorflow 2.4.1). Also, you will need Python libraries like ***Numpy*** and ***Matplotlib*** and the ***Tensorflow Probability*** library. Finally, a instruction list is shown below.

1. Install Python 3.X.
2. Install Numpy.
3. Install Matplotlib.
4. [Install Tensorflow 2.4.X](https://www.tensorflow.org/install)
5. [Install Tensorflow Probability](https://www.tensorflow.org/probability/install)
6. [Install A-Connect Library](https://github.com/onchipuis/A-Connect) or...
7. Include the A-Connect git submodule in (/Tensorflow/aconnect)


### Table of Contents
1. [aconnect](/Tensorflow/aconnect): This folder contains the final library for the methodology (git submodule from https://github.com/onchipuis/A-Connect).
2. [Networks](/Tensorflow/Networks): All the networks used for testing purpose. Contains the scripts for training and testing (e.g., MC simulations). Please see local readme.
3. [Networks/Layers](/Tensorflow/Networks/Layers): Some additional layers created for testing (e.g., DVA). Please see local readme.
4. [Networks/Models](/Tensorflow/Networks/Models): Obtained NN architecture models after training.
5. [Results](/Tensorflow/Results): This folder contains the results of the Monte Carlo and other simulations.

