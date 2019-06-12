# README #
This package is a PyTorch port of the original **S**pike **LAY**er **E**rror **R**eassignment (**SLAYER**) framework for backpropagation based spiking neural networks (SNNs) learning.
The original implementation is in C++ with CUDA and CUDNN. 
It is available at [https://bitbucket.org/bamsumit/slayer](https://bitbucket.org/bamsumit/slayer) .

A brief introduction of the method is in the following video.

[![](http://img.youtube.com/vi/JGdatqqci5o/0.jpg)](http://www.youtube.com/watch?v=JGdatqqci5o "")

The base description of the framework has been published in [NeurIPS 2018](https://nips.cc/Conferences/2018/Schedule?showEvent=11157).
The final paper is available [here](http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf).
The arXiv preprint is available [here](https://arxiv.org/abs/1810.08646).

## Citation ##
Sumit Bam Shrestha and Garrick Orchard. "SLAYER: Spike Layer Error Reassignment in Time." 
In _Advances in Neural Information Processing Systems_, pp. 1417-1426. 2018.

```bibtex
@InCollection{Shrestha2018,
  author    = {Shrestha, Sumit Bam and Orchard, Garrick},
  title     = {{SLAYER}: Spike Layer Error Reassignment in Time},
  booktitle = {Advances in Neural Information Processing Systems 31},
  publisher = {Curran Associates, Inc.},
  year      = {2018},
  editor    = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
  pages     = {1419--1428},
  url       = {http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf},
}
```

## What is this repository for? ##

* For learning weight (delay learning not yet implemented) parameters of a multilayer spiking neural network.
* Natively handles multiple spikes in each layer and error backpropagation through the layers. 
* Version 0.1

## Requirements
Python 3 with the following packages installed:

* PyTorch (tested with version 1.0.1.post2)
* numpy
* pyyaml

A **CUDA** enabled **GPU** is required for training any model.
No plans on CPU only implementation yet.
The software has been tested with CUDA libraries version 9.2 and GCC 7.3.0 on Ubuntu 18.04

## Installation
The repository includes C++ and CUDA code that has to be compiled and installed before it can be used from Python, download the repository and run the following command to do so:

`python setup.py install`

To test the installation:

`cd test`

`python -m unittest`

## Documentation
The complete documentation is available at [https://bamsumit.github.io/slayerPytorch](https://bamsumit.github.io/slayerPytorch) .

## Examples
Example implementations can be found inside Examples folder.

* Run example MLP implementation

	`>>> python nmnistMLP.py`
	
* Run example CNN implementation
	
	`>>> python nmnistCNN.py`

## Contribution
* By [Sumit Bam Shrestha](mailto:bam_sumit@hotmail.com).
* This work builds on initial implementation by [Luca Della VEDOVA](mailto:lucadellavr@gmail.com).

## Contact
For queries contact [Sumit](mailto:bam_sumit@hotmail.com).

### License & Copyright ###
Copyright 2018 Sumit Bam Shrestha
SLAYER-PyTorch is free software: you can redistribute it and/or modoify it under the terms of 
GNU General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

SLAYER-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License SLAYER.
If not, see http://www.gnu.org/licenses/.
