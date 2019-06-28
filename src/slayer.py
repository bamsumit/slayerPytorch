import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# import slayer_cuda
import slayerCuda
# import matplotlib.pyplot as plt

# # Consider dictionary for easier iteration and better scalability
# class yamlParams(object):
# 	'''
# 	This class reads yaml parameter file and allows dictionary like access to the members.
	
# 	Usage:

# 	.. code-block:: python
		
# 		import slayerSNN as snn
# 		netParams = snn.params('path_to_yaml_file')	# OR
# 		netParams = slayer.yamlParams('path_to_yaml_file')

# 		netParams['training']['learning']['etaW'] = 0.01
# 		print('Simulation step size        ', netParams['simulation']['Ts'])
# 		print('Spiking neuron time constant', netParams['neuron']['tauSr'])
# 		print('Spiking neuron threshold    ', netParams['neuron']['theta'])

# 		netParams.save('filename.yaml')
# 	'''
# 	def __init__(self, parameter_file_path):
# 		with open(parameter_file_path, 'r') as param_file:
# 			self.parameters = yaml.safe_load(param_file)

# 	# Allow dictionary like access
# 	def __getitem__(self, key):
# 		return self.parameters[key]

# 	def __setitem__(self, key, value):
# 		self.parameters[key] = value

# 	def save(self, filename):
# 		with open(filename, 'w') as f:
# 			yaml.dump(self.parameters, f)

# class spikeLayer():
class spikeLayer(torch.nn.Module):
	'''
	This class defines the main engine of SLAYER.
	It provides necessary funcitons for describing a SNN layer.
	The input to output connection can be fully-connected, convolutional, or aggregation (pool)
	It also defines the psp operation and spiking mechanism of a spiking neuron in the layer.

	**Important:** It assumes all the tensors that are being processed are 5 dimensional. 
	(Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
	The user must make sure that an input of correct dimension is supplied.

	*If the layer does not have spatial dimension, the neurons can be distributed along either
	Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
	It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
	dimension one.*

	Arguments:
		* ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
			.. code-block:: python

				neuron:
				    type:     SRMALPHA	# neuron type
				    theta:    10	# neuron threshold
				    tauSr:    10.0	# neuron time constant
				    tauRef:   1.0	# neuron refractory time constant
				    scaleRef: 2		# neuron refractory response scaling (relative to theta)
				    tauRho:   1		# spike function derivative time constant (relative to theta)
				    scaleRho: 1		# spike function derivative scale factor
		* ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
			.. code-block:: python

				simulation:
				    Ts: 1.0
				    tSample: 300
				    nSample: 12		
		* ``fullRefKernel`` (``bool``, optional): high resolution refractory kernel (the user shall not use it in practice)  
	
	Usage:

	>>> snnLayer = slayer.spikeLayer(neuronDesc, simulationDesc)
	'''
	def __init__(self, neuronDesc, simulationDesc, fullRefKernel = False):
		super(spikeLayer, self).__init__()
		self.neuron = neuronDesc
		self.simulation = simulationDesc
		self.fullRefKernel = fullRefKernel
		
		# self.srmKernel = self.calculateSrmKernel()
		# self.refKernel = self.calculateRefKernel()
		self.register_buffer('srmKernel', self.calculateSrmKernel())
		self.register_buffer('refKernel', self.calculateRefKernel())
		
	def calculateSrmKernel(self):
		srmKernel = self._calculateAlphaKernel(self.neuron['tauSr'])
		# TODO implement for different types of kernels
		return torch.tensor(srmKernel)
		# return torch.tensor( self._zeroPadAndFlip(srmKernel)) # to be removed later when custom cuda code is implemented
		
	def calculateRefKernel(self):
		if self.fullRefKernel:
			refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -2 * self.neuron['theta'], EPSILON = 0.0001)
			# This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
		else:
			refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -2 * self.neuron['theta'])
		
		# TODO implement for different types of kernels
		return torch.tensor(refKernel)
		
	def _calculateAlphaKernel(self, tau, mult = 1, EPSILON = 0.01):
		# could be made faster... NOT A PRIORITY NOW
		eps = []
		# tauSr = self.neuron['tauSr']
		for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
			epsVal = mult * t / tau * math.exp(1 - t / tau)
			if abs(epsVal) < EPSILON and t > tau:
				break
			eps.append(epsVal)
		return eps
	
	def _zeroPadAndFlip(self, kernel):
		if (len(kernel)%2) == 0: kernel.append(0)
		prependedZeros = np.zeros((len(kernel) - 1))
		return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()
		
	def psp(self, spike):
		'''
		Applies psp filtering to spikes.
		The output tensor dimension is same as input.

		Arguments:
			* ``spike``: input spike tensor.

		Usage:

		>>> filteredSpike = snnLayer.psp(spike)
		'''
		return _pspFunction.apply(spike, self.srmKernel, self.simulation['Ts'])

	def pspLayer(self):
		'''
		Returns a function that can be called to apply psp filtering to spikes.
		The output tensor dimension is same as input.
		The initial psp filter corresponds to the neuron psp filter.
		The psp filter is learnable.
		NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.
		
		Usage:
		
		>>> pspLayer = snnLayer.pspLayer()
		>>> filteredSpike = pspLayer(spike)
		'''
		return _pspLayer(self.srmKernel, self.simulation['Ts'])

	def replicateInTime(self, input, mode='nearest'):
		Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
		N, C, H, W = input.shape
		# output = F.pad(input.reshape(N, C, H, W, 1), pad=(Ns-1, 0, 0, 0, 0, 0), mode='replicate')
		if mode == 'nearest':
			output = F.interpolate(input.reshape(N, C, H, W, 1), size=(H, W, Ns), mode='nearest')
		return output
	
	def dense(self, inFeatures, outFeatures, weightScale=10):	# default weight scaling of 10
		'''
		Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
		It behaves similar to ``torch.nn.Linear`` applied for each time instance.

		Arguments:
			* ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
			  dimension of input features (Width, Height, Channel) that represents the number of input neurons.
			* ``outFeatures`` (``int``): number of output neurons.

		Usage:
		
		>>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
		>>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
		>>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
		'''
		return _denseLayer(inFeatures, outFeatures, weightScale)	
		
	def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100):	# default weight scaling of 100
		'''
		Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
		It behaves same as ``torch.nn.conv2d`` applied for each time instance.

		Arguments:
			* ``inChannels`` (``int``): number of channels in input
			* ``outChannels`` (``int``): number of channls produced by convoluion
			* ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
			* ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
			* ``padding`` (``int`` or tuple of two ints):	zero-padding added to both sides of the input. Default: 0
			* ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
			* ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1

		The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

		- a single ``int`` -- in which case the same value is used for the height and width dimension
		- a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
		  and the second `int` for the width dimension

		Usage:

		>>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
		>>> output = conv(input)           # must have 2 channels
		'''
		return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale) 
		
	def pool(self, kernelSize, stride=None, padding=0, dilation=1):
		'''
		Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
		It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

		Arguments:
			* ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
			* ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
			* ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
			* ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
			
		The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

		- a single ``int`` -- in which case the same value is used for the height and width dimension
		- a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
		  and the second `int` for the width dimension

		Usage:

		>>> pool = snnLayer.pool(4) # 4x4 pooling
		>>> output = pool(input)
		'''
		return _poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation)

	def dropout(self, p=0.5, inplace=False):
		'''
		Returns a function that can be called to apply dropout layer to the input tensor.
		It behaves similar to ``torch.nn.Dropout``.
		However, dropout over time dimension is preserved, i.e.
		if a neuron is dropped, it remains dropped for entire time duration.

		Arguments:
			* ``p``: dropout probability.
			* ``inplace`` (``bool``): inplace opeartion flag.

		Usage:

		>>> drop = snnLayer.dropout(0.2)
		>>> output = drop(input)
		'''
		return _dropoutLayer(p, inplace)

	def delayShift(self, input, delay, Ts=1):
		'''
		Applies delay in time dimension (assumed to be the last dimension of the tensor) of the input tensor.
		The autograd backward link is established as well.

		Arguments:
			* ``input``: input Torch tensor.
			* ``delay`` (``float`` or Torch tensor): amount of delay to apply.
			  Same delay is applied to all the inputs if ``delay`` is ``float`` or Torch tensor of size 1.
			  If the Torch tensor has size more than 1, its dimension  must match the dimension of input tensor except the last dimension.
			* ``Ts``: sampling time of the delay. Default is 1.
		
		Usage:

		>>> delayedInput = slayer.delayShift(input, 5)
		'''
		return _delayFunctionNoGradient.apply(input, delay, Ts)

	def delay(self, inputSize):
		'''
		Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
		The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
		The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
		The delay values are not clamped to zero.
		To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

		Arguments:
			* ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
			  If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

		Usage:

		>>> delay = snnLayer.delay((C, H, W))
		>>> delayedSignal = delay(input)

		Always clamp the delay after ``optimizer.step()``.

		>>> optimizer.step()
		>>> delay.delay.data.clamp_(0)	
		'''
		return _delayLayer(inputSize, self.simulation['Ts'])
	
	# def applySpikeFunction(self, membranePotential):
	# 	return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

	def spike(self, membranePotential):
		'''
		Applies spike function and refractory response.
		The output tensor dimension is same as input.
		``membranePotential`` will reflect spike and refractory behaviour as well.

		Arguments:
			* ``membranePotential``: subthreshold membrane potential.

		Usage:

		>>> outSpike = snnLayer.spike(membranePotential)
		'''
		return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

class _denseLayer(nn.Conv3d):
	def __init__(self, inFeatures, outFeatures, weightScale=1):
		'''
		'''
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
		
		super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

		if weightScale != 1:	
			self.weight = torch.nn.Parameter(weightScale * self.weight)	# scale the weight if needed
			# print('In dense, using weightScale of', weightScale)

	
	def forward(self, input):
		'''
		'''
		return F.conv3d(input, 
						self.weight, self.bias, 
						self.stride, self.padding, self.dilation, self.groups)

class _convLayer(nn.Conv3d):
	'''
	'''
	def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1):
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

		super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

		if weightScale != 1:	
			self.weight = torch.nn.Parameter(weightScale * self.weight)	# scale the weight if needed
			# print('In conv, using weightScale of', weightScale)

	def foward(self, input):
		'''
		'''
		return F.conv3d(input, 
						self.weight, self.bias, 
						self.stride, self.padding, self.dilation, self.groups)

class _poolLayer(nn.Conv3d):
	'''
	'''
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
		
		super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)	

		# set the weights to 1.1*theta and requires_grad = False
		self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)
		# print('In pool layer, weight =', self.weight.cpu().data.numpy().flatten(), theta)


	def forward(self, input):
		'''
		'''
		device = input.device
		dtype  = input.dtype

		# add necessary padding for odd spatial dimension
		if input.shape[2]%2 != 0:
			input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], 1, input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
		if input.shape[3]%2 != 0:
			input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], 1, input.shape[4]), dtype=dtype).to(device)), 3)

		dataShape = input.shape

		result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
						  self.weight, self.bias, 
						  self.stride, self.padding, self.dilation)
		# print(result.shape)
		return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))

class _dropoutLayer(nn.Dropout3d):
	'''
	'''
	# def __init__(self, p=0.5, inplace=False):
	# 	super(_dropoutLayer, self)(p, inplace)

	'''
	'''
	def forward(self, input):
		inputShape = input.shape
		return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
						   self.p, self.training, self.inplace).reshape(inputShape)

class _pspLayer(nn.Conv3d):
	'''
	'''
	def __init__(self, filter, Ts):
		inChannels  = 1
		outChannels = 1
		kernel      = (1, 1, torch.numel(filter))

		self.Ts = Ts

		super(_pspLayer, self).__init__(inChannels, outChannels, kernel, bias=False) 

		# print(filter)
		# print(np.flip(filter.cpu().data.numpy()).reshape(self.weight.shape)) 
		# print(torch.tensor(np.flip(filter.cpu().data.numpy()).copy()))

		flippedFilter = torch.tensor(np.flip(filter.cpu().data.numpy()).copy()).reshape(self.weight.shape)

		self.weight = torch.nn.Parameter(flippedFilter.to(self.weight.device), requires_grad = True)

		self.pad = torch.nn.ConstantPad3d(padding=(torch.numel(filter)-1, 0, 0, 0, 0, 0), value=0)

	def forward(self, input):
		'''
		'''
		inShape = input.shape
		inPadded = self.pad(input.reshape((inShape[0], 1, 1, -1, inShape[-1])))
		# print((inShape[0], 1, 1, -1, inShape[-1]))
		# print(input.reshape((inShape[0], 1, 1, -1, inShape[-1])).shape)
		# print(inPadded.shape)
		output = F.conv3d(inPadded, self.weight) * self.Ts
		return output.reshape(inShape)

class _spikeFunction(torch.autograd.Function):
	'''
	'''
	@staticmethod
	def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
		'''
		'''
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
			
		# (membranePotential, spikes) = slayer_cuda.get_spikes_cuda(membranePotential,
		# 														  torch.empty_like(membranePotential),	# tensor for spikes
		# 														  refractoryResponse,
		# 														  threshold,
		# 														  Ts)
		spikes = slayerCuda.getSpikes(membranePotential, refractoryResponse, threshold, Ts)
		
		pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
		# pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho']                   , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
		pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
		threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
		# torch.cuda.synchronize()
		
		# if device != oldDevice: torch.cuda.set_device(oldDevice)
		# torch.cuda.device(oldDevice)
		
		return spikes
		
	@staticmethod
	def backward(ctx, gradOutput):
		'''
		'''
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

class _pspFunction(torch.autograd.Function):
	'''
	'''
	@staticmethod
	def forward(ctx, spike, filter, Ts):
		device = spike.device
		dtype  = spike.dtype
		psp = slayerCuda.conv(spike, filter, Ts)
		Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(filter, Ts)
		return psp

	@staticmethod
	def backward(ctx, gradOutput):
		'''
		'''
		(filter, Ts) = ctx.saved_tensors
		gradInput = slayerCuda.corr(gradOutput, filter, Ts)
		if filter.requires_grad is False:
			gradFilter = None
		else:
			gradFilter = None
			pass
			
		return gradInput, gradFilter, None

class _delayLayer(nn.Module):
	'''
	'''
	def __init__(self, inputSize, Ts):
		super(_delayLayer, self).__init__()

		if type(inputSize) == int:
			inputChannels = inputSize
			inputHeight   = 1
			inputWidth    = 1
		elif len(inputSize) == 3:
			inputChannels = inputSize[0]
			inputHeight   = inputSize[1]
			inputWidth    = inputSize[2]
		else:
			raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

		self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
		# self.delay = torch.nn.Parameter(torch.empty((inputChannels, inputHeight, inputWidth)), requires_grad=True)
		# print('delay:', torch.empty((inputChannels, inputHeight, inputWidth)))
		self.Ts = Ts

	def forward(self, input):

		return _delayFunction.apply(input, self.delay, self.Ts)

class _delayFunction(torch.autograd.Function):
	'''
	'''
	@staticmethod
	def forward(ctx, input, delay, Ts):
		'''
		'''
		device = input.device
		dtype  = input.dtype
		output = slayerCuda.shift(input, delay.data, Ts)
		Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(output, delay.data, Ts)
		return output

	@staticmethod
	def backward(ctx, gradOutput):
		'''
		'''
		# autograd tested and verified
		(output, delay, Ts) = ctx.saved_tensors
		diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
		outputDiff = slayerCuda.conv(output, diffFilter, 1)
		# the conv operation should not be scaled by Ts. 
		# As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
		gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
		# no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

		return slayerCuda.shift(gradOutput, -delay, Ts), gradDelay, None

class _delayFunctionNoGradient(torch.autograd.Function):
	'''
	'''
	@staticmethod
	def forward(ctx, input, delay, Ts=1):
		'''
		'''
		device = input.device
		dtype  = input.dtype
		output = slayerCuda.shift(input, delay, Ts)
		Ts     = torch.autograd.Variable(torch.tensor(Ts   , device=device, dtype=dtype), requires_grad=False)
		delay  = torch.autograd.Variable(torch.tensor(delay, device=device, dtype=dtype), requires_grad=False)
		ctx.save_for_backward(delay, Ts)
		return output

	@staticmethod
	def backward(ctx, gradOutput):
		'''
		'''
		(delay, Ts) = ctx.saved_tensors
		return slayerCuda.shift(gradOutput, -delay, Ts), None, None
