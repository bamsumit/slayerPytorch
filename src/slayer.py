import math
import numpy as np
import torch
import torch.nn as nn

def EPSILON():
	return 0.01

class spikeLayer:
	def __init__(self, neuronDesc, simulationDesc, device=torch.device('cuda')):
		self.neuron = neuronDesc
		self.simulation = simulationDesc
		self.device = device
		self.srmKernel = self.calculateSrmKernel()
		self.refKernel = self.calculateRefKernel()
		
	def calculateSrmKernel(self):
		srmKernel = self._calculateAlphaKernel()
		# TODO implement for different types of kernels
		# return torch.tensor(srmKernel, device = self.device)
		return torch.tensor( self._zeroPadAndFlip(srmKernel), device = self.device) # to be removed later when custom cuda code is implemented
		
	def calculateRefKernel(self):
		refKernel = self._calculateAlphaKernel(mult = -2 * self.neuron['theta'])
		# TODO implement for different types of kernels
		return torch.tensor(refKernel, device = self.device)
		
	def _calculateAlphaKernel(self, mult = 1):
		# could be made faster... NOT A PRIORITY NOW
		eps = []
		tauSr = self.neuron['tauSr']
		for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
			epsVal = mult * t / tauSr * math.exp(1 - t / tauSr)
			if abs(epsVal) < abs(EPSILON()) and t > tauSr:
				break
			eps.append(epsVal)
		return eps
	
	def _zeroPadAndFlip(self, kernel):
		if (len(kernel)%2) == 0: kernel.append(0)
		prependedZeros = np.zeros((len(kernel) - 1))
		return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()
		
	def applySrmKernel(self, spike):
		return nn.functional.conv3d(spike, 
									self.srmKernel.reshape((1, 1, 1, 1, len(self.srmKernel))),
									padding = (0, 0, int( self.srmKernel.shape[0] / 2 ) ))
	
	def dense():
		pass
		
	def conv():
		pass
		
	def pool():
		pass
		
	def spike():
		pass

class denseLayer(nn.Conv3d):
	def __init__(self, inFeatures, outFeatures):
		kernel = ()
	pass