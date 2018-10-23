This package is a Pytorch port of the original Spike LAYer Error Reassignment framework for backpropagation based spiking neural networks (SNNs) learning.

## Requirements
* pytorch, tested with version 0.4.1
* numpy
* pyyaml
* unittest

A CUDA enabled GPU is required for training any model.

## Installation
The repository includes C++ and Cuda code that has to be compiled and installed before it can be used from Python, download the repository and run the following command to do so:

`python setup.py install`

## Examples
An example of how to train the network for classification purposes, on the NMNIST dataset, can be found in the nminst_net.py file. Make sure to change the FILES_DIR variable to indicate the folder where you downloaded the NMNIST dataset.

A .yaml file is used to load and customise most parameters (i.e. neuron thresholds), an example can be found under test/test_files/NMNISTsmall/parameters.yaml. This file will also be used for the NMNIST example code.