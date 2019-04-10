import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import slayer_cuda
# import matplotlib.pyplot as plt

# Consider dictionary for easier iteration and better scalability
class yamlParams(object):
	def __init__(self, parameter_file_path):
		with open(parameter_file_path, 'r') as param_file:
			self.parameters = yaml.load(param_file)

	# Allow dictionary like access
	def __getitem__(self, key):
		return self.parameters[key]

	def __setitem__(self, key, value):
		self.parameters[key] = value

class spikeLayer:
	def __init__(self, neuronDesc, simulationDesc, device=torch.device('cuda'), dtype=torch.float32, fullRefKernel = False):
		self.neuron = neuronDesc
		self.simulation = simulationDesc
		self.device = device
		self.dtype = dtype
		self.fullRefKernel = fullRefKernel
		
		self.srmKernel = self.calculateSrmKernel()
		self.refKernel = self.calculateRefKernel()
		
	def calculateSrmKernel(self):
		srmKernel = self._calculateAlphaKernel()
		# TODO implement for different types of kernels
		# return torch.tensor(srmKernel, device = self.device)
		return torch.tensor( self._zeroPadAndFlip(srmKernel), device = self.device, dtype = self.dtype) # to be removed later when custom cuda code is implemented
		
	def calculateRefKernel(self):
		if self.fullRefKernel:
			refKernel = self._calculateAlphaKernel(mult = -2 * self.neuron['theta'], EPSILON = 0.0001)
			# This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
		else:
			refKernel = self._calculateAlphaKernel(mult = -2 * self.neuron['theta'])
		
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
		
	def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1):
		return convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups).to(self.device)
		
	def pool(self, kernelSize, stride=None, padding=0, dilation=1):
		return poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation).to(self.device)
		
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

class convLayer(nn.Conv3d):
	def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1):
		inChannels = inFeatures
		outChannels = outFeatures
		
		# kernel
		if type(kernelSize) == int:
			kernel = (kernelSize, kernelSize, 1)
		elif len(kernelSize) == 2:
			kernel = (kernelSize[0], kernelSize[1], 1)
		else:
			raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

		# stride
		if type(stride) == int:
			stride = (stride, stride, 1)
		elif len(stride) == 2:
			stride = (stride[0], stride[1], 1)
		else:
			raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

		# padding
		if type(padding) == int:
			padding = (padding, padding, 0)
		elif len(padding) == 2:
			padding = (padding[0], padding[1], 0)
		else:
			raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

		# dilation
		if type(dilation) == int:
			dilation = (dilation, dilation, 1)
		elif len(dilation) == 2:
			dilation = (dilation[0], dilation[1], 1)
		else:
			raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

		# groups
		# no need to check for groups. It can only be int

		# print('inChannels :', inChannels)
		# print('outChannels:', outChannels)
		# print('kernel     :', kernel, kernelSize)
		# print('stride     :', stride)
		# print('padding    :', padding)
		# print('dilation   :', dilation)
		# print('groups     :', groups)

		super(convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

	def foward(self, input):
		return F.conv3d(input, 
						self.weight, self.bias, 
						self.stride, self.padding, self.dilation, self.groups)

class poolLayer(nn.Conv3d):
	def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1):
		# kernel
		if type(kernelSize) == int:
			kernel = (kernelSize, kernelSize, 1)
		elif len(kernelSize) == 2:
			kernel = (kernelSize[0], kernelSize[1], 1)
		else:
			raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
		
		# stride
		if stride is None:
			stride = kernel
		elif type(stride) == int:
			stride = (stride, stride, 1)
		elif len(stride) == 2:
			stride = (stride[0], stride[1], 1)
		else:
			raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

		# padding
		print(padding)
		if type(padding) == int:
			padding = (padding, padding, 0)
		elif len(padding) == 2:
			padding = (padding[0], padding[1], 0)
		else:
			raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

		# dilation
		if type(dilation) == int:
			dilation = (dilation, dilation, 1)
		elif len(dilation) == 2:
			dilation = (dilation[0], dilation[1], 1)
		else:
			raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

		# print('theta      :', theta)
		# print('kernel     :', kernel, kernelSize)
		# print('stride     :', stride)
		# print('padding    :', padding)
		# print('dilation   :', dilation)
		
		super(poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)	

		# set the weights to 1.1*theta and requires_grad = False
		self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)

	def forward(self, input):
		dataShape = input.shape
		result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
						  self.weight, self.bias, 
						  self.stride, self.padding, self.dilation)
		return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))
						
class spikeFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
		device = membranePotential.device
		dtype  = membranePotential.dtype
		threshold      = neuron['theta']
		oldDevice = torch.cuda.current_device()

		# if device != oldDevice: torch.cuda.set_device(device)
		# torch.cuda.device(3)

		# spikeTensor = torch.empty_like(membranePotential)

		# print('membranePotential  :', membranePotential .device)
		# print('spikeTensor        :', spikeTensor       .device)
		# print('refractoryResponse :', refractoryResponse.device)
			
		(membranePotential, spikes) = slayer_cuda.get_spikes_cuda(membranePotential,
																  torch.empty_like(membranePotential),	# tensor for spikes
																  refractoryResponse,
																  threshold,
																  Ts)
		
		pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
		# pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho']                   , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
		pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
		threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
		torch.cuda.synchronize()
		
		# if device != oldDevice: torch.cuda.set_device(oldDevice)
		# torch.cuda.device(oldDevice)
		
		return spikes
		
	@staticmethod
	def backward(ctx, gradOutput):
		(membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
		spikePdf = pdfScale / pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)

		# return gradOutput, None, None, None # This seems to work better!
		return gradOutput * spikePdf, None, None, None
		# plt.figure()
		# plt.plot(gradOutput[0,5,0,0,:].cpu().data.numpy())
		# print   (gradOutput[0,0,0,0,:].cpu().data.numpy())
		# plt.plot(membranePotential[0,0,0,0,:].cpu().data.numpy())
		# plt.plot(spikePdf         [0,0,0,0,:].cpu().data.numpy())
		# print   (spikePdf         [0,0,0,0,:].cpu().data.numpy())
		# plt.show()
		# return gradOutput * spikePdf, None, None, None

