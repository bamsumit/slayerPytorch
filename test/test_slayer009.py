import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn
import slayerCuda

device = torch.device('cuda:1')
Ts = 0.1

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)

(N, C, H, W, D) = (5, 10, 20, 30, 500)

delay = slayer.delay((C, H, W)).to(device)

inTensor = torch.randn((N, C, H, W, D)).to(device)

delay.delay.data = torch.rand((delay.delay.data.shape)).to(device) * 2 - 1
outTensor = slayerCuda.shift(inTensor, delay.delay.data, Ts)

def checkShift(N, C, H, W, verbose=False):
	shift = int( (delay.delay.data[C, H, W] / Ts).item() )
	if shift > 0:
		error = torch.norm(inTensor[N, C, H, W, :-shift] - outTensor[N, C, H, W, shift:]).item()
	elif shift == 0:
		error = torch.norm(inTensor[N, C, H, W, :] - outTensor[N, C, H, W, :]).item()
	else:
		error = torch.norm(inTensor[N, C, H, W, -shift:] - outTensor[N, C, H, W, :shift]).item()

	if verbose is True:
		print('delay\n', shift, delay.delay.data[C, H, W])
		print('input\n', inTensor[N, C, H, W, :])
		print('output\n', outTensor[N, C, H, W, :])
		# print('output\n', outTensor[N, C, H, W, :] - inTensor[N, C, H, W, :])
		print('error :', error)

	return error

failed = False
for n in range(N):
	for c in range(C):
		for h in range(H):
			for w in range(W):
				error = checkShift(n, c, h, w)
				# print (n, c, h, w, error)
				if error > 1e-6: failed = True
				if failed is True: break
			if failed is True: break
		if failed is True: break
	if failed is True: break

print('Shift Error :', checkShift(n, c, h, w))