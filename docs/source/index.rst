.. SLAYER PyTorch documentation master file, created by
   sphinx-quickstart on Tue May  7 15:14:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SLAYER PyTorch's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   slayerSNN.rst
   slayer.rst
   spikeClassifier.rst
   spikeLoss.rst
   spikeIO.rst
   learningStats.rst
   optimizer.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Usage:
------

>>> import slayerSNN as snn

* The **spike-layer** module will be available as ``snn.layer``.
* The **yaml-parameter** module will be availabe as ``snn.params``.
* The **spike-loss** module will be available as ``snn.loss``.
* The **spike-classifier** module will be available as ``snn.predict``.
* The **spike-IO** module will be available as ``snn.io``.

Example:
--------
The SNN parameters are stored in a yaml file. 
The structure of the yaml file follows the same hierarchy as the 
`C++ SLAYER framework
<https://bitbucket.org/bamsumit/slayer>`_ (see Network Description)

.. code-block:: python
   :linenos:

   import slayerSNN as snn
   # other imports and definitions

   class Network(torch.nn.Module):
      def __init__(self, netParams, device=device):
         super(Network, self).__init__()
         # initialize slayer
         slayer = snn.layer(netParams['neuron'], netParams['simulation'], device=device)
         self.sl = slayer
         # define network functions
         self.conv1 = slayer.conv(2, 16, 5, padding=1)
         self.conv2 = slayer.conv(16, 32, 3, padding=1)
         self.conv3 = slayer.conv(32, 64, 3, padding=1)
         self.pool1 = slayer.pool(2)
         self.pool2 = slayer.pool(2)
         self.fc1   = slayer.dense((8, 8, 64), 10)

      def forward(self, spikeInput):
         spikeLayer1 = self.sl.spike(self.conv1(self.sl.psp(spikeInput)))  # 32, 32, 16
         spikeLayer2 = self.sl.spike(self.pool1(self.sl.psp(spikeLayer1))) # 16, 16, 16
         spikeLayer3 = self.sl.spike(self.conv2(self.sl.psp(spikeLayer2))) # 16, 16, 32
         spikeLayer4 = self.sl.spike(self.pool2(self.sl.psp(spikeLayer3))) #  8,  8, 32
         spikeLayer5 = self.sl.spike(self.conv3(self.sl.psp(spikeLayer4))) #  8,  8, 64
         spikeOut    = self.sl.spike(self.fc1  (self.sl.psp(spikeLayer5))) #  10
         return spikeOut
         
   # network
   net = Network(snn.params('path to yaml file'))

   # cost function
   error = snn.loss(netParams)

   # dataloader not shown. input and target are assumed to be available
   output = net.forward(input)
   loss = error.numSpikes(output, target)

**Important:** It is assumed that all the tensors that are being processed are 5 dimensional. 
(Batch, Channels, Height, Width, Time) or ``NCHWT`` format. 
The user must make sure that an input of correct dimension is supplied.
 
*If the layer does not have spatial dimension, the neurons can be distributed along either
Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
dimension one.*
