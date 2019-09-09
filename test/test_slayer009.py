import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn
import slayerCuda

device = torch.device('cuda')
Ts = 0.1

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)

# (N, C, H, W, D) = (2, 5, 6, 7, 50)
(N, C, H, W, D) = (5, 10, 20, 30, 500)
# Uncomment this to test large neuron sizes
# (N, C, H, W, D) = (5, 16, 128, 128, 500)

delay = slayer.delay((C, H, W)).to(device)

inTensor = torch.randn((N, C, H, W, D)).to(device)

def checkShift(inTensor, outTensor, shift, verbose=False):
	if shift > 0:
		error = torch.norm(inTensor[:-shift] - outTensor[shift:]).item()
	elif shift == 0:
		error = torch.norm(inTensor - outTensor).item()
	else:
		error = torch.norm(inTensor[-shift:] - outTensor[:shift]).item()

	if verbose is True:
		print('input\n', inTensor)
		print('output\n', outTensor)
		# print('output\n', outTensor[N, C, H, W, :] - inTensor[N, C, H, W, :])
		print('error :', error)

	return error

class TestDelay(unittest.TestCase):
	def testShiftPerNeuron(self):

		delay.delay.data = torch.rand((delay.delay.data.shape)).to(device) * 2 - 1
		outTensor = slayerCuda.shift(inTensor, delay.delay.data, Ts)

		netError = 0

		for n in range(N):
			for c in range(C):
				for h in range(H):
					for w in range(W):
						shift = int( (delay.delay.data[c, h, w] / Ts).item() )
						error = checkShift(inTensor[n, c, h, w], outTensor[n, c, h, w], shift)
						# print (n, c, h, w, error)
						netError += error

		# print('Shift (per neuron) Error:', checkShift(n, c, h, w, shift))
		self.assertEqual(netError, 0, 'Shift (per neuron) error must be zero.')

	def testShift(self):
		delay = 4

		outTensor = slayer.delayShift(inTensor, delay)

		netError = 0
		
		for n in range(N):
			for c in range(C):
				for h in range(H):
					for w in range(W):
						error = checkShift(inTensor[n, c, h, w], outTensor[n, c, h, w], delay)
						# print (n, c, h, w, error)
						netError += error

		# print('Shift (individual) Error:', checkShift(n, c, h, w, 4))
		self.assertEqual(netError, 0, 'Shift error must be zero.')

if __name__ == '__main__':
	unittest.main()