import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn


###############################################################################
# To test the correctness of convolution operation and pooling operation ######

device = torch.device('cuda') 

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation'], device=device)

# conv operation test
# (N, C, H, W, D) = (1, 1, 8, 8, 1)
(N, C, H, W, D) = (4, 8, 100, 100, 500)
K = 5

conv = slayer.conv(C, 1, K)
inTensor = torch.randn((N, C, H, W, D)).to(device)

outGT = torch.zeros((N, 1, H-K+1, W-K+1, D)).to(device)

weight = conv.weight.cpu().data.numpy().reshape((C, K, K))

for i in range(K):
	for j in range(K):
		temp = inTensor[:, :, j:H-K+1+j, i:W-K+1+i, :]
		for k in range(C):
			outGT += weight[k, j, i] * temp[:,k,...].reshape((N,1,H-K+1, W-K+1, D))

out = conv(inTensor)

error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)

print('Conv Error :', error)	

# print(inTensor.cpu().data)
# print(conv.weight.cpu().data)
# print(out.cpu().data)
# print(outGt.cpu().data)

# pooling operation
pool = slayer.pool(2)
inTensor = torch.randn((4, 8, 100, 100, 500)).to(device)

outGT = ( inTensor[:, :, 0::2, 0::2, :] + \
  		  inTensor[:, :, 0::2, 1::2, :] + \
  		  inTensor[:, :, 1::2, 0::2, :] + \
  		  inTensor[:, :, 1::2, 1::2, :] ) * 1.1 * netParams['neuron']['theta']
out   = pool(inTensor)

error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)

print('Pool Error :', error)
