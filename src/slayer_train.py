import math
import numpy as np
from data_reader import SlayerParams

class SlayerTrainer(object):

	def __init__(self, net_params):
		self.net_params = net_params

	def calculate_srm_kernel(self):
		return self._calculate_srm_kernel(self.net_params['sr_params']['tau'], self.net_params['sr_params']['epsilon'], 
			self.net_params['t_end'], self.net_params['t_res'])

	def _calculate_srm_kernel(self, tau, epsilon, t_end, t_res):
		srm_kernel = []
		for t in range(0, t_end, t_res):
			srm_val = t / tau * math.exp(1 - t / tau)
			# Make sure we break after the peak
			if srm_val < epsilon and t > tau:
				break
			srm_kernel.append(srm_val)
		# Convert to numpy array and reshape in a shape compatible for 3d convolution
		srm_kernel = np.asarray(srm_kernel)
		np.reshape(srm_kernel, (1,1,len(srm_kernel)))
		return srm_kernel