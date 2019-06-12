import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn

###############################################################################
# To test dropout operation ###################################################

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

device = torch.device('cuda')

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)

(N, C, H, W, D) = (4, 8, 100, 100, 500)

drop = slayer.dropout(p=0.1).to(device)
if verbose is True:
	print('Dropout Probability:', drop.p)
	print('Inplace:', drop.inplace)

inTensor = torch.randn((N, C, H, W, D)).to(device)
outTensor = drop(inTensor)

if verbose is True:
	print('in shape :', inTensor.shape)
	print('out shape:', outTensor.shape)

inSum  = inTensor. sum(-1, keepdim=True).cpu().data.numpy().flatten()
outSum = outTensor.sum(-1, keepdim=True).cpu().data.numpy().flatten()

class TestDropout(unittest.TestCase):
	def test(self):
		checkStatus = True
		for (i, o) in zip(inSum, outSum):
			if abs(i - o*(1-drop.p))>1e-3 and  abs(o)> 1e-6:	
				print(o, i, i/o)
				checkStatus=False
				break

		# print('Result ' + ('Passed' if checkStatus is True else 'Failed'))
		self.assertTrue(checkStatus, 'Neurons not dropped out must match in value.')

if __name__ == '__main__':
	unittest.main()
