import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn

# Testing trainable psp

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

# Select device to run code on
device = torch.device('cuda')

# Initialization
netParams = snn.params('test_files/snnData/network.yaml')

N    = 5 # number of batch
Ts   = netParams['simulation']['Ts']
Ns   = int(netParams['simulation']['tSample'] / Ts)
Nin  = int(netParams['layer'][1]['dim'])

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)
pspLayer = slayer.pspLayer().to(device)

# Generate spikes
# input spikes
spikeIn = torch.zeros((N, Nin, 1, 1, Ns)).to(device)
spikeIn[torch.rand((N, Nin, 1, 1, Ns)) > 0.8] = 1/Ts

pspGT = slayer.psp(spikeIn)
psp   = pspLayer(spikeIn)


class TestPspLayer(unittest.TestCase):
	def test(self):
		error = torch.norm(psp - pspGT) / torch.numel(psp)
		if verbose is True:	print('error =', error)
		self.assertTrue(error < 1e-6, 'Result from Psp function and Psp layer must match.')

# Testing time replication
class TestReplicate(unittest.TestCase):
	def test(self):
		input = torch.rand((1, 1, 5, 6)).to(device)
		output = slayer.replicateInTime(input)
		if verbose is True:
			print(input)
			print(output.shape)
			print(output[0, 0, :, :, 5])
		for t in range(output.shape[-1]):
			error = torch.norm(input - output[...,t]) / torch.numel(input)
			self.assertTrue(error < 1e-6, 'Result must match with input at each time step.')

if __name__ == '__main__':
	unittest.main()

