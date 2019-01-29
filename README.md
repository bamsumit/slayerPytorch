This package is a Pytorch port of the original Spike LAYer Error Reassignment framework for backpropagation based spiking neural networks (SNNs) learning.

This work builds on initial implementation by [Luca Della VEDOVA](mailto:lucadellavr@gmail.com).

## Requirements
Python 3 with the following packages installed:

* pytorch, tested with version 0.4.1
* numpy
* pyyaml
* unittest

A CUDA enabled GPU is required for training any model.
The software has been tested with CUDA libraries version 9.2 and GCC 7.3.0 on Ubuntu 18.04

## Installation
The repository includes C++ and Cuda code that has to be compiled and installed before it can be used from Python, download the repository and run the following command to do so:

`python setup.py install`

## Examples
An example of how to train the network for classification purposes, on the NMNIST dataset, can be found in the nminst_net.py file. Make sure to change the FILES_DIR variable to indicate the folder where you downloaded the NMNIST dataset.

A .yaml file is used to load and customise most parameters (i.e. neuron thresholds), an example can be found under test/test_files/NMNISTsmall/parameters.yaml. This file will also be used for the NMNIST example code.

## Contribution
This work builds on initial implementation by [Luca Della VEDOVA](mailto:lucadellavr@gmail.com).

## Contact
For queries contact [Sumit](mailto:bam_sumit@hotmail.com).