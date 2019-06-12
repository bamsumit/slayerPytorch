import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn


###############################################################################
# To test the correctness of convolution operation and pooling operation ######

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

device = torch.device('cuda') 

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)

# conv operation test
# (N, C, H, W, D) = (1, 1, 8, 8, 1)
(N, C, H, W, D) = (4, 8, 100, 100, 500)
K = 5

conv = slayer.conv(C, 1, K).to(device)
inTensor = torch.randn((N, C, H, W, D)).to(device)


weight = conv.weight.cpu().data.numpy().reshape((C, K, K))

class TestConvolution(unittest.TestCase):
	def test(self):
		outGT = torch.zeros((N, 1, H-K+1, W-K+1, D)).to(device)
		for i in range(K):
			for j in range(K):
				temp = inTensor[:, :, j:H-K+1+j, i:W-K+1+i, :]
				for k in range(C):
					outGT += weight[k, j, i] * temp[:,k,...].reshape((N,1,H-K+1, W-K+1, D))

		out = conv(inTensor)

		error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)

		# print('Conv Error :', error)
		self.assertTrue(error < 1e-6, 'Conv result (out) must match outGT.')
		

# print(inTensor.cpu().data)
# print(conv.weight.cpu().data)
# print(out.cpu().data)
# print(outGt.cpu().data)

# pooling operation
pool = slayer.pool(2).to(device)
inTensor = torch.randn((4, 8, 100, 100, 500)).to(device)

class TestPooling(unittest.TestCase):
	def test(self):
		outGT = ( inTensor[:, :, 0::2, 0::2, :] + \
		  		  inTensor[:, :, 0::2, 1::2, :] + \
		  		  inTensor[:, :, 1::2, 0::2, :] + \
		  		  inTensor[:, :, 1::2, 1::2, :] ) * 1.1 * netParams['neuron']['theta']
		out   = pool(inTensor)

		error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)

		# print('Pool Error :', error)
		self.assertTrue(error < 1e-6, 'Pool result (out) must match outGT.')
	
	def testOddDimension(self):
		# odd pooling
		if verbose is True: print('Testing Odd dimension pooling')
		inTensor = torch.randn((4, 8, 101, 101, 500)).to(device)
		out = pool(inTensor)
			
		# print(out.shape)
		# If this executes, it works
		self.assertTrue(True, 'Odd dimension pooling failed.')

if __name__ == '__main__':
	unittest.main()
