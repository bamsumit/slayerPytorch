import math
import numpy as np
from data_reader import SlayerParams
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlayerTrainer(object):

	def __init__(self, net_params):
		self.net_params = net_params
		# Data type used for membrane potentials, weights
		self.data_type = np.float32

	def calculate_srm_kernel(self):
		single_kernel = self._calculate_srm_kernel(self.net_params['sr_params']['tau'], self.net_params['sr_params']['epsilon'], 
			self.net_params['t_end'], self.net_params['t_s'])
		concatenated_srm =  self._concatenate_srm_kernel(single_kernel, self.net_params['input_channels'])
		return torch.from_numpy(concatenated_srm)

	# Generate kernels that will act on a single channel (0 outside of diagonals)
	def _concatenate_srm_kernel(self, kernel, n_channels):
		eye_tensor = np.reshape(np.eye(n_channels, dtype = self.data_type), (n_channels, n_channels, 1, 1, 1))
		return kernel * eye_tensor

	def _calculate_srm_kernel(self, tau, epsilon, t_end, t_s):
		srm_kernel = []
		for t in range(0, t_end, t_s):
			srm_val = t / tau * math.exp(1 - t / tau)
			# Make sure we break after the peak
			if srm_val < epsilon and t > tau:
				break
			srm_kernel.append(srm_val)
		# Make sure the kernel is odd size (for convolution)
		if ((len(srm_kernel) % 2) == 0) : srm_kernel.append(0)
		# Convert to numpy array and reshape in a shape compatible for 3d convolution
		srm_kernel = np.asarray(srm_kernel, dtype = self.data_type)
		# Prepend length-1 zeros to make the convolution filter causal
		prepended_zeros = np.zeros((len(srm_kernel)-1,), dtype = self.data_type)
		srm_kernel = np.flip(np.concatenate((prepended_zeros, srm_kernel)))
		return srm_kernel.reshape((1,1,len(srm_kernel)))
		# Convert to pytorch tensor
		
	def apply_srm_kernel(self, input_spikes, srm):
		out = F.conv3d(input_spikes, srm, padding=(0,0,int(srm.shape[4]/2))) * self.net_params['t_s']
		return out