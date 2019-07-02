import sys, os

CURRENT_SRC_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_SRC_DIR + "/../../slayerPyTorch/src")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import slayer
import slayerCuda
import slayerLoihiCuda
from quantizeParams import quantizeWeights

class spikeLayer(slayer.spikeLayer):
	def __init__(self, neuronDesc, simulationDesc):
		if neuronDesc['type'] == 'LOIHI':
			neuronDesc['theta'] = neuronDesc['vThMant'] * 2**6
		
		super(spikeLayer, self).__init__(neuronDesc, simulationDesc)

		self.maxPspKernel = torch.max(self.srmKernel).cpu().data.item()
		print('Max PSP kernel:', self.maxPspKernel)
		print('Scaling neuron[scaleRho] by Max PSP Kernel @slayerLoihi')
		neuronDesc['scaleRho'] /= self.maxPspKernel
		
	def calculateSrmKernel(self):
		srmKernel = self._calculateLoihiPSP()
		return torch.tensor(srmKernel)
		
	def calculateRefKernel(self, SCALE=1000):
		refKernel = self._calculateLoihiRefKernel(SCALE)
		return torch.tensor(refKernel)
		
	def _calculateLoihiPSP(self):
		# u = [0]
		# v = [0]
		u = []
		v = []
		u.append( 1 << (6 + self.neuron['wgtExp'] + 1) ) # +1 to compensate for weight resolution of 2 for mixed synapse mode
		v.append( u[-1] ) # we do not consider bias in slayer
		while v[-1] > 0:
			uNext = ( ( u[-1] * ( (1<<12) - self.neuron['iDecay']) ) >> 12 )
			vNext = ( ( v[-1] * ( (1<<12) - self.neuron['vDecay']) ) >> 12 ) + uNext # again, we do not consider bias in slayer
			u.append(uNext)
			v.append(vNext)

		return  [float(x)/2 for x in v]	# scale by half to compensate for 1 in the initial weight

	def _calculateLoihiRefKernel(self, SCALE=1000):
		absoluteRefKernel = np.ones(self.neuron['refDelay']) * (-SCALE * self.neuron['theta'])
		absoluteRefKernel[0] = 0
		relativeRefKernel = [ self.neuron['theta'] ]
		while relativeRefKernel[-1] > 0:
			nextRefKernel = ( relativeRefKernel[-1] * ( (1<<12) - self.neuron['vDecay']) ) >> 12 
			relativeRefKernel.append(nextRefKernel)
		refKernel = np.concatenate( (absoluteRefKernel, -2 * np.array(relativeRefKernel) ) ).astype('float32')
		return refKernel

	def spikeLoihi(self, weightedSpikes):
		return _spike.apply(weightedSpikes, self.srmKernel, self.neuron, self.simulation['Ts'])

	def spikeLoihiFull(self, weightedSpikes):
		return _spike.loihi(weightedSpikes, self.neuron, self.simulation['Ts'])

	def dense(self, inFeatures, outFeatures, weightScale=100, quantize=True):
		return _denseLayer(inFeatures, outFeatures, weightScale, quantize)

	def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, quantize=True):
		return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, quantize)
	
	def pool(self, kernelSize, stride=None, padding=0, dilation=1):
		requiredWeight = quantizeWeights.apply(torch.tensor(1.1 * self.neuron['theta'] / self.maxPspKernel), 2).cpu().data.item()
		# print('Required pool layer weight =', requiredWeight)
		return slayer._poolLayer(requiredWeight/ 1.1, # to compensate for maxPsp
						  kernelSize, stride, padding, dilation)

	def getVoltage(self, membranePotential):
		Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
		voltage = membranePotential.reshape((-1, Ns)).cpu().data.numpy()
		return np.where(voltage <= -500*self.neuron['theta'], self.neuron['theta'] + 1, voltage)

class _denseLayer(slayer._denseLayer):
	def __init__(self, inFeatures, outFeatures, weightScale=1, quantize=True):
		self.quantize=True
		super(_denseLayer, self).__init__(inFeatures, outFeatures, weightScale)

	def forward(self, input):
		if self.quantize is True:
			return F.conv3d(input, 
							quantizeWeights.apply(self.weight, 2), self.bias,
							self.stride, self.padding, self.dilation, self.groups)
		else:
			return F.conv3d(input, 
							self.weight, self.bias,
							self.stride, self.padding, self.dilation, self.groups)

class _convLayer(slayer._convLayer):
	def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, quantize=True):
		self.quantize=True
		super(_convLayer, self).__init__(inFeatures, outFeatures, kernelSize, stride, padding, dilation, groups, weightScale)
 
	def forward(self, input):
		if self.quantize is True:
			return F.conv3d(input, 
							quantizeWeights.apply(self.weight, 2), self.bias,
							self.stride, self.padding, self.dilation, self.groups)
		else:
			return F.conv3d(input, 
							self.weight, self.bias, 
							self.stride, self.padding, self.dilation, self.groups)


class _spike(torch.autograd.Function):
	'''
	'''
	@staticmethod
	def loihi(weightedSpikes, neuron, Ts):
		iDecay = neuron['iDecay']
		vDecay = neuron['vDecay']
		theta  = neuron['theta']
		# wScale = 1 << (6 + neuron['wgtExp'])
		wgtExp = neuron['wgtExp']

		if weightedSpikes.dtype == torch.int32:
			Ts = 1
		
		spike, voltage, current = slayerLoihiCuda.getSpikes(weightedSpikes * Ts, wgtExp, theta, iDecay, vDecay)

		return spike/Ts, voltage, current

	@staticmethod
	def forward(ctx, weightedSpikes, srmKernel, neuron, Ts):
		device = weightedSpikes.device
		dtype  = weightedSpikes.dtype
		pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
		pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
		threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
		Ts              = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
		srmKernel       = torch.autograd.Variable(srmKernel.clone().detach(), requires_grad=False)

		
		spike, voltage, current = _spike.loihi(weightedSpikes, neuron, Ts)

		ctx.save_for_backward(voltage, threshold, pdfTimeConstant, pdfScale, srmKernel, Ts)
		return spike

	@staticmethod
	def backward(ctx, gradOutput):
		(membranePotential, threshold, pdfTimeConstant, pdfScale, srmKernel, Ts) = ctx.saved_tensors
		spikePdf = pdfScale / pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)

		return slayerCuda.corr(gradOutput * spikePdf, srmKernel, Ts), None, None, None
