import numpy as np
import torch
 
class spikeClassifier:
	@staticmethod
	def getClass(spike):
		numSpikes = torch.sum(spike, 4, keepdim=True).cpu().data.numpy()
		return np.argmax(numSpikes.reshape((numSpikes.shape[0], -1)), 1)