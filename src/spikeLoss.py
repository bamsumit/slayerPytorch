import math
import numpy as np
import torch
import torch.nn as nn
from slayer import spikeLayer

class spikeLoss(torch.nn.Module):	
	'''
	This class defines different spike based loss modules that can be used to optimize the SNN.

	Usage:

	>>> error = spikeLoss.spikeLoss(networkDescriptor)
	>>> error = spikeLoss.spikeLoss(errorDescriptor, neuronDesc, simulationDesc)
	'''
	def __init__(self, errorDescriptor, neuronDesc, simulationDesc):
		super(spikeLoss, self).__init__()
		self.neuron = neuronDesc
		self.simulation = simulationDesc
		self.errorDescriptor = errorDescriptor
		self.slayer = spikeLayer(neuronDesc, simulationDesc)
		# self.dtype = slayer.dtype
		# self.device = slayer.device
		self.psp = self.slayer.psp()

	def __init__(self, networkDescriptor):
		super(spikeLoss, self).__init__()
		self.neuron = networkDescriptor['neuron']
		self.simulation = networkDescriptor['simulation']
		self.errorDescriptor = networkDescriptor['training']['error']
		self.slayer = spikeLayer(self.neuron, self.simulation)
		# self.dtype = slayer.dtype
		# self.device = slayer.device
		self.psp = self.slayer.psp()
	
	def spikeTime(self, spikeOut, spikeDesired):
		'''
		Calculates spike loss based on spike time.
		The loss is similar to van Rossum distance between output and desired spike train.

		.. math::

			E = \int_0^T \left( \varepsilon * (output - desired)\right)(t) ^2 \text{d}t

		Arguments:
			* ``spikeOut`` (``torch.tensor``): spike tensor
			* ``spikeDesired`` (``torch.tensor``): desired spike tensor

		Usage:

		>>> loss = error.spikeTime(spikeOut, spikeDes)
		'''
		# Tested with autograd, it works
		assert self.errorDescriptor['type'] == 'SpikeTime', "Error type is not SpikeTime"
		error = self.psp(spikeOut - spikeDesired) 
		return 1/2 * torch.sum(error**2) * self.simulation['Ts']
	
	def numSpikes(self, spikeOut, desiredClass, numSpikesScale=1):
		'''
		Calculates spike loss based on number of spikes within a `target region`.
		The `target region` and `desired spike count` is specified in ``error.errorDescriptor['tgtSpikeRegion']``
		Any spikes outside the target region are penalized with ``error.spikeTime`` loss..

		.. math::

			e(t) &=
			\begin{cases}
			\frac{acutalSpikeCount - desiredSpikeCount}{targetRegionLength} & \text{for }t \in targetRegion\\
			\left(\varepsilon * (output - desired)\right)(t) & \text{otherwise}
			\end{cases}
			\\
			E &= \int_0^T e(t)^2 \text{d}t
			\end{align}

		Arguments:
			* ``spikeOut`` (``torch.tensor``): spike tensor
			* ``desiredClass`` (``torch.tensor``): one-hot encoded desired class tensor. Time dimension should be 1 and rest of the tensor dimensions should be same as ``spikeOut``.

		Usage:

		>>> loss = error.numSpikes(spikeOut, target)
		'''

		# Tested with autograd, it works
		assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not NumSpikes"
		# desiredClass should be one-hot tensor with 5th dimension 1
		tgtSpikeRegion = self.errorDescriptor['tgtSpikeRegion']
		tgtSpikeCount  = self.errorDescriptor['tgtSpikeCount']
		startID = np.rint( tgtSpikeRegion['start'] / self.simulation['Ts'] ).astype(int)
		stopID  = np.rint( tgtSpikeRegion['stop' ] / self.simulation['Ts'] ).astype(int)
		
		actualSpikes = torch.sum(spikeOut, 4, keepdim=True).cpu().detach().numpy() * self.simulation['Ts']
		desiredSpikes = np.where(desiredClass.cpu() == True, tgtSpikeCount[True], tgtSpikeCount[False])
		# print('actualSpikes :', actualSpikes.flatten())
		# print('desiredSpikes:', desiredSpikes.flatten())
		errorSpikeCount = (actualSpikes - desiredSpikes) / (stopID - startID) * numSpikesScale
		targetRegion = np.zeros(spikeOut.shape)
		targetRegion[:,:,:,:,startID:stopID] = 1;
		spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)
		
		error = self.psp(spikeOut - spikeDesired)
		error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)
		
		return 1/2 * torch.sum(error**2) * self.simulation['Ts']
	
	def probSpikes(spikeOut, spikeDesired, probSlidingWindow = 20):
		assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not ProbSpikes"
		pass
