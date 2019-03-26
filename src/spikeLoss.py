import math
import numpy as np
import torch
import torch.nn as nn
from slayer import spikeLayer

class spikeLoss:	
	def __init__(self, slayer):
		self.neuron = slayer.neuron
		self.simulation = slayer.simulation
		self.dtype = slayer.dtype
		self.device = slayer.device
		self.psp = slayer.psp()

	# def __init__(self, neuronDesc, simulationDesc, device=torch.device('cuda')):
	# 	self.neuron = neuronDesc
	# 	self.simulation = simulationDesc
	# 	self.device = device
	# 	slayer = spikeLayer(self.neuron, self.simulation, device=self.device)
	# 	self.psp = slayer.psp()
		
	def spikeTime(self, spikeOut, spikeDesired):
		error = self.psp(spikeOut - spikeDesired) 
		return 1/2 * torch.sum(error**2) * self.simulation['Ts']
	
	def numSpikes(self, spikeOut, spikeDesired):
		error = torch.zeros((spikeOut.shape))
		# t_valid = self.simulation['tSample']
		# error[:,:,:,:,0:t_valid] = (torch.sum(spikeOut[:,:,:,:,0:t_valid], 4, keepdim=True) - spikeDesired) / (t_valid / self.simulation['Ts'])
		error = (torch.sum(spikeOut, 4, keepdim=True) - spikeDesired) / (spikeOut.shape[-1] / self.simulation['Ts'])
		return 1/2 * torch.sum(error ** 2) * self.simulation['Ts']
	
	def probSpikes(spikeOut, spikeDesired, probSlidingWindow = 20):
		pass