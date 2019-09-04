import numpy as np
import torch
 
class spikeClassifier:
	'''
	It provides classification modules for SNNs.
	All the functions it supplies are static and can be called without making an instance of the class.
	'''
	@staticmethod
	def getClass(spike):
		'''
		Returns the predicted class label.
		It assignes single class for the SNN output for the whole simulation runtime.

		Usage:

		>>> predictedClass = spikeClassifier.getClass(spikeOut)
		'''
		numSpikes = torch.sum(spike, 4, keepdim=True).cpu()
		return torch.max(numSpikes.reshape((numSpikes.shape[0], -1)), 1)[1]
		# numSpikes = torch.sum(spike, 4, keepdim=True).cpu().data.numpy()
		# return np.argmax(numSpikes.reshape((numSpikes.shape[0], -1)), 1)