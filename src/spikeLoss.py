import math
import numpy as np
import torch
import torch.nn as nn
from slayer import spikeLayer

class spikeLoss:	
	def __init__(self, slayer, errorDescriptor):
		self.neuron = slayer.neuron
		self.simulation = slayer.simulation
		self.errorDescriptor = errorDescriptor
		self.dtype = slayer.dtype
		self.device = slayer.device
		self.psp = slayer.psp()
	
	def spikeTime(self, spikeOut, spikeDesired):
		# Tested with autograd, it works
		assert self.errorDescriptor['type'] == 'SpikeTime', "Error type is not SpikeTime"
		error = self.psp(spikeOut - spikeDesired) 
		return 1/2 * torch.sum(error**2) * self.simulation['Ts']
	
	def numSpikes(self, spikeOut, desiredClass):
		# Tested with autograd, it works
		assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not NumSpikes"
		# desiredClass should be one-hot tensor with 5th dimension 1
		tgtSpikeRegion = self.errorDescriptor['tgtSpikeRegion']
		tgtSpikeCount  = self.errorDescriptor['tgtSpikeCount']
		startID = np.rint( tgtSpikeRegion['start'] / self.simulation['Ts'] ).astype(int)
		stopID  = np.rint( tgtSpikeRegion['stop' ] / self.simulation['Ts'] ).astype(int)
		
		actualSpikes = torch.sum(spikeOut, 4, keepdim=True).cpu().detach().numpy()
		desiredSpikes = np.where(desiredClass == True, tgtSpikeCount[True], tgtSpikeCount[False])
		errorSpikeCount = (actualSpikes - desiredSpikes) / (stopID - startID)
		targetRegion = np.zeros(spikeOut.shape)
		targetRegion[:,:,:,:,startID:stopID] = 1;
		spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)
		
		error = self.psp(spikeOut - spikeDesired)
		error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)
		
		return 1/2 * torch.sum(error**2) * self.simulation['Ts']
	
	def probSpikes(spikeOut, spikeDesired, probSlidingWindow = 20):
		assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not ProbSpikes"
		pass
