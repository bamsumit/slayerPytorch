import math
import numpy as np
from data_reader import SlayerParams
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlayerNet(nn.Module):

    def __init__(self, net_params, weights_init = [4,4]):
        super(SlayerNet, self).__init__()
        self.net_params = net_params
        self.trainer = SlayerTrainer(net_params)
        self.srm = self.trainer.calculate_srm_kernel()
        self.ref = self.trainer.calculate_ref_kernel()
        # Emulate a fully connected 250 -> 25
        self.fc1 = nn.Conv3d(1, 25, (1,250,1), bias=False)
        nn.init.normal_(self.fc1.weight, mean=0, std=weights_init[0])
        # Emulate a fully connected 25 -> 1
        self.fc2 = nn.Conv3d(1, 1, (1,25,1), bias=False)
        nn.init.normal_(self.fc2.weight, mean=0, std=weights_init[1])

    def forward(self, x):
        # Apply srm to input spikes
        x = self.trainer.apply_srm_kernel(x, self.srm)
        # Linear + activation
        x = SpikeFunc.apply(self.fc1(x), self.net_params, self.ref, self.net_params['af_params']['sigma'][0])
        # Apply srm to middle layer spikes
        x = self.trainer.apply_srm_kernel(x.view(1,1,1,25,501), self.srm)
        # # Apply second layer
        x = SpikeFunc.apply(self.fc2(x), self.net_params, self.ref, self.net_params['af_params']['sigma'][1])
        return x

class SpikeFunc(torch.autograd.Function):

	@staticmethod
	def forward(ctx, multiplied_activations, net_params, ref, sigma):
		# Calculate membrane potentials
		(potentials, spikes) = SpikeFunc.calculate_membrane_potentials(multiplied_activations, net_params, ref, sigma)
		scale = torch.autograd.Variable(torch.Tensor([net_params['pdf_params']['scale']]), requires_grad=False)
		tau = torch.autograd.Variable(torch.Tensor([net_params['pdf_params']['tau']]), requires_grad=False)
		theta = torch.autograd.Variable(torch.Tensor([net_params['af_params']['theta']]), requires_grad=False)
		ctx.save_for_backward(potentials, theta, tau, scale)
		return spikes

	@staticmethod
	def backward(ctx, grad_output):
		(membrane_potentials, theta, tau, scale) = ctx.saved_tensors
		# Don't return any gradient for parameters
		return (grad_output * SpikeFunc.calculate_pdf(membrane_potentials, theta, tau, scale), None, None, None)

	@staticmethod
	def apply_weights(activations, weights):
		applied = F.conv3d(activations, weights)
		return applied

	# Input is potentials after matrix multiplication
	@staticmethod
	def calculate_membrane_potentials(potentials, net_params, ref, sigma):
		# Need float32 to do convolution later
		spikes = torch.zeros(potentials.shape, dtype=torch.float32)
		ref_length = len(ref)
		# Iterate over timestamps, NOTE, check if iterating in this dimension is a bottleneck
		for p in range(potentials.shape[-1]):
			ts_pots = potentials[:,:,:,:,p]
			spike_positions = (ts_pots > net_params['af_params']['theta']).to(dtype=torch.float32)
			spike_values = spike_positions / net_params['t_s']
			# Assign output spikes
			spikes[:,:,:,:,p] = spike_values
			num_spikes = torch.sum(spike_positions, dim=(1,2,3))
			have_spike_response = ref * (1 + sigma * (num_spikes - 1))
			no_spike_response = ref * (sigma * num_spikes)
			# Now iterate over neurons, apply refractory response
			for n_id in range(spikes.shape[1]):
				# Make sure our refractory response doesn't overshoot the total time
				resp_length = min(potentials.shape[-1] - p, ref_length)
				if spike_positions[:,n_id,0,0] > 0:
					# Have spike here
					potentials[:,n_id,0,0,p:p+resp_length] += have_spike_response[0:resp_length]
				else:
					# Didn't have a spike
					potentials[:,n_id,0,0,p:p+resp_length] += no_spike_response[0:resp_length]
			# print(num_spikes)
		return (potentials, spikes)

	@staticmethod
	def calculate_pdf(membrane_potentials, theta, tau, scale):
		pdf = scale / tau * torch.exp(-abs(membrane_potentials - theta) / tau)
		return pdf

class SlayerTrainer(object):

	def __init__(self, net_params):
		self.net_params = net_params
		# Data type used for membrane potentials, weights
		self.data_type = np.float32

	def calculate_srm_kernel(self):
		single_kernel = self._calculate_srm_kernel(self.net_params['sr_params']['mult'], self.net_params['sr_params']['tau'],
			self.net_params['sr_params']['epsilon'], self.net_params['t_end'], self.net_params['t_s'])
		concatenated_srm =  self._concatenate_srm_kernel(single_kernel, self.net_params['input_channels'])
		return torch.from_numpy(concatenated_srm)

	# Generate kernels that will act on a single channel (0 outside of diagonals)
	def _concatenate_srm_kernel(self, kernel, n_channels):
		eye_tensor = np.reshape(np.eye(n_channels, dtype = self.data_type), (n_channels, n_channels, 1, 1, 1))
		return kernel * eye_tensor

	def _calculate_srm_kernel(self, mult, tau, epsilon, t_end, t_s):
		srm_kernel = self._calculate_eps_func(mult, tau, epsilon, t_end, t_s)
		# Make sure the kernel is odd size (for convolution)
		if ((len(srm_kernel) % 2) == 0) : srm_kernel.append(0)
		# Convert to numpy array and reshape in a shape compatible for 3d convolution
		srm_kernel = np.asarray(srm_kernel, dtype = self.data_type)
		# Prepend length-1 zeros to make the convolution filter causal
		prepended_zeros = np.zeros((len(srm_kernel)-1,), dtype = self.data_type)
		srm_kernel = np.flip(np.concatenate((prepended_zeros, srm_kernel)))
		return srm_kernel.reshape((1,1,len(srm_kernel)))
		# Convert to pytorch tensor

	def _calculate_eps_func(self, mult, tau, epsilon, t_end, t_s):
		eps_func = []
		for t in np.arange(0, t_end, t_s):
			srm_val = mult * t / tau * math.exp(1 - t / tau)
			# Make sure we break after the peak
			if abs(srm_val) < abs(epsilon) and t > tau:
				break
			eps_func.append(srm_val)
		return eps_func
		
	def calculate_ref_kernel(self):
		ref_kernel = self._calculate_eps_func(self.net_params['ref_params']['mult'], self.net_params['ref_params']['tau'],
			self.net_params['ref_params']['epsilon'], self.net_params['t_end'], self.net_params['t_s'])
		return torch.FloatTensor(ref_kernel)

	def apply_srm_kernel(self, input_spikes, srm):
		out = F.conv3d(input_spikes, srm, padding=(0,0,int(srm.shape[4]/2))) * self.net_params['t_s']
		return out

	def calculate_error_spiketrain(self, a, des_a):
		return a - des_a

	def calculate_l2_loss(self, a, des_a):
		return torch.sum(self.calculate_error_spiketrain(a, des_a) ** 2) / 2 * self.net_params['t_s']