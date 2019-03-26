import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import slayer_cuda

class spikeLayer:
	def __init__(self, neuronDesc, simulationDesc, device=torch.device('cuda'), dtype=torch.float32):
		self.neuron = neuronDesc
		self.simulation = simulationDesc
		self.device = device
		self.dtype = dtype
		
		self.srmKernel = self.calculateSrmKernel()
		self.refKernel = self.calculateRefKernel()
		
	def calculateSrmKernel(self):
		srmKernel = self._calculateAlphaKernel()
		# TODO implement for different types of kernels
		# return torch.tensor(srmKernel, device = self.device)
		return torch.tensor( self._zeroPadAndFlip(srmKernel), device = self.device, dtype = self.dtype) # to be removed later when custom cuda code is implemented
		
	def calculateRefKernel(self):
		refKernel = self._calculateAlphaKernel(mult = -2 * self.neuron['theta'])
		# refKernel = self._calculateAlphaKernel(mult = -2 * self.neuron['theta'], EPSILON = 0.0001)
		# This gives the exact precision as MATLAB implementation, however, it is expensive
		#
		# TODO implement for different types of kernels
		return torch.tensor(refKernel, device = self.device, dtype = self.dtype)
		
	def _calculateAlphaKernel(self, mult = 1, EPSILON = 0.01):
		# could be made faster... NOT A PRIORITY NOW
		eps = []
		tauSr = self.neuron['tauSr']
		for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
			epsVal = mult * t / tauSr * math.exp(1 - t / tauSr)
			if abs(epsVal) < EPSILON and t > tauSr:
				break
			eps.append(epsVal)
		return eps
	
	def _zeroPadAndFlip(self, kernel):
		if (len(kernel)%2) == 0: kernel.append(0)
		prependedZeros = np.zeros((len(kernel) - 1))
		return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()
		
	def applySrmKernel(self, spike):
		spikeShape = spike.shape
		return nn.functional.conv3d(spike.reshape( (spikeShape[0], 1, spikeShape[1] * spikeShape[2], spikeShape[3], spikeShape[4]) ), 
									self.srmKernel.reshape((1, 1, 1, 1, len(self.srmKernel))),
									padding = (0, 0, int( self.srmKernel.shape[0] / 2 ) )).reshape(spikeShape) * self.simulation['Ts']

	def psp(self):
		return lambda spike : self.applySrmKernel(spike)
	
	def dense(self, inFeatures, outFeatures):
		return denseLayer(inFeatures, outFeatures).to(self.device)
		
	def conv():
		pass
		
	def pool():
		pass
		
	def spike(self):
		return lambda membranePotential : spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

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
		# print('Kernel Dimension:', kernel)
		# print('Input Channels  :', inChannels)
		
		if type(outFeatures) == int:
			outChannels = outFeatures
		else:
			raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
		# print('Output Channels :', outChannels)
		
		super(denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
	
	def forward(self, input):
		return F.conv3d(input, 
						self.weight, self.bias, 
						self.stride, self.padding, self.dilation, self.groups)
						
class spikeFunction(torch.autograd.Function): # NOT TESTED YET
	@staticmethod
	def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
		device = membranePotential.device
		dtype  = membranePotential.dtype
		threshold      = neuron['theta']
		oldDevice = torch.cuda.current_device()
		
		# if device != oldDevice: torch.cuda.set_device(device)
			
		(membranePotential, spikes) = slayer_cuda.get_spikes_cuda(membranePotential,
																  torch.empty_like(membranePotential),	# tensor for spikes
																  refractoryResponse,
																  threshold,
																  Ts)
		# TODO change it to return spikes only. The membranePotential should be changed intrinsically
		pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho'], device=device, dtype=dtype), requires_grad=False)
		pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho']  , device=device, dtype=dtype), requires_grad=False)
		threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']   , device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
		
		# if device != oldDevice: torch.cuda.set_device(oldDevice)
		
		return spikes
		
	@staticmethod
	def backward(ctx, gradOutput):
		(membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
		spikePdf = pdfScale / pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)
		# return gradOutput * spikePdf, None, None, None
		return gradOutput, None, None, None