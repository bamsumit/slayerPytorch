import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
	
	def dense(self, inFeatures, outFeatures):
		return denseLayer(inFeatures, outFeatures)
		
	def conv():
		pass
		
	def pool():
		pass
		
	def spike(self, membranePotential):
		return spikeFunction(membranePotential)

class denseLayer(nn.Conv3d):
	def __init__(self, inFeatures, outFeatures):
		# extract information for kernel and inChannels
		if type(inFeatures) == int:
			kernel = (1, 1, 1)
			inChannels = inFeatures 
		elif len(inFeatures) == 2:
			kernel = (inFeatures[1], inFeatures[0], 1)
			inChannels = 1
		elif len(inFeatures) == 3:
			kernel = (inFeatures[1], inFeatures[0], 1)
			inChannels = inFeatures[2]
		else:
			raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
		print('Kernel Dimension:', kernel)
		print('Input Channels  :', inChannels)
		
		if type(outFeatures) == int:
			outChannels = outFeatures
		else:
			raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
		print('Output Channels :', outChannels)
		
		super(denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
	
	def forward(self, input):
		return F.conv3d(input, 
						self.weight, self.bias, 
						self.stride, self.padding, self.dilation, self.groups)
						
class spikeFunction(torch.autograd.Function):
	def __init__(self, )
		pass
		
	@staticmethod
	def forward(ctx, membranePotential)
		pass
		
	@staticmethod
	def backward(ctx, gradOutput):
		pass