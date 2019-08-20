import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + '/../src')

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn

# Testing filter banks

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

# Select device to run code on
device = torch.device('cuda')

# Initialization
netParams = snn.params('test_files/snnData/network.yaml')

N  = 5 # number of batch
Ts = netParams['simulation']['Ts']
Ns = int(netParams['simulation']['tSample'] / Ts)
C  = 8
H  = 16
W  = 16

nFilter = 5
slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)
pspFilter = slayer.pspFilter(nFilter=nFilter, filterLength=slayer.srmKernel.numel()).to(device)

# Generate spikes
# input spikes
spikeIn = torch.zeros((N, C, H, W, Ns)).to(device)
spikeIn[torch.rand((N, C, H, W, Ns)) > 0.8] = 1/Ts

flippedFilter = torch.tensor(np.flip(slayer.srmKernel.cpu().data.numpy()).copy())
pspFilter.weight.data = flippedFilter.reshape((1, 1, 1, 1, -1)).repeat((5, 1, 1, 1, 1)).to(device)

pspGT = slayer.psp(spikeIn)
pspFB = pspFilter(spikeIn)

class TestPspFilter(unittest.TestCase):
	def testShape(self):
		self.assertEqual(pspGT.shape[1] * nFilter, pspFB.shape[1], 'Channel dimension must be scaled by number of filters.')

	def testValues(self):
		error = 0
		for i in range(nFilter):
			error += torch.norm(pspFB[:, i*8:(i+1)*8, :, :, :] - pspGT) / torch.numel(pspGT)
		if verbose is True:	print('error =', error)
		self.assertTrue(error < 1e-6, 'Result from Psp function and Psp filter bank must match.')

