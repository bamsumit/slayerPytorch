import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn
import slayerCuda

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

# A gradient logger to probe autgrad gradients
class gradLog(torch.autograd.Function):
	data = []

	@staticmethod
	def forward(ctx, input):
		return input

	@staticmethod
	def backward(ctx, gradOutput):
		gradLog.data.append(gradOutput)
		return gradOutput

	@staticmethod
	def reset():
		gradLog.data = []


# Select device to run code on
device = torch.device('cuda')

# Initialization
netParams = snn.params('test_files/snnData/network.yaml')

N    = 5 # number of batch
Ts   = netParams['simulation']['Ts']
Ns   = int(netParams['simulation']['tSample'] / Ts)
Nin  = int(netParams['layer'][1]['dim'])
Nout = int(netParams['layer'][2]['dim'])

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)
fc1    = slayer.dense(Nin, Nout).to(device)
delay  = slayer.delay((Nin, 1, 1)).to(device)

# Generate spikes
# input spikes
spikeIn = torch.zeros((N, Nin, 1, 1, Ns)).to(device)
spikeIn[torch.rand((N, Nin, 1, 1, Ns)) > 0.8] = 1/Ts

# desired spikes
spikeDes = torch.zeros((N, Nout, 1, 1, Ns)).to(device)
spikeDes[torch.rand((N, Nout, 1, 1, Ns)) > 0.8] = 1/Ts

# AutoGrad 1:
# first Psp opeartion followed by delay
gradLog.reset()
psp = slayer.psp(spikeIn)
pspDelayed = delay(psp)
uOut = fc1(gradLog.apply(pspDelayed))
spikeOut = slayer.spike(gradLog.apply(uOut))

# loss
error = snn.loss(netParams).to(device)
loss  = error.spikeTime(spikeOut, spikeDes)

loss.backward()

# Custom calculation of delay gradient
deltaRec = gradLog.data[0]
errorRec = gradLog.data[1]
# filter to differentiate sinal in time dimension
diffFilter = torch.tensor([1, -1], dtype=torch.float).to(device)/Ts
# psp derivative signal
dpspDelayed_dt = slayerCuda.conv(pspDelayed, diffFilter, 1)
# delay graident integration (According to the formula)
delayGrad = -torch.sum(errorRec * dpspDelayed_dt, [0, -1], keepdim=True).reshape((Nin, 1, 1)) * Ts

class TestAutoGrad1(unittest.TestCase):
	def test(self):
		# print('CustomDelayGradient - autoGrad1:', torch.norm(delayGrad - delay.delay.grad).item())
		# self.assertEqual(torch.norm(delayGrad - delay.delay.grad).item(), 0, 'CustomDelayGradient and AutoGrad1 results must match.')
		self.assertTrue(torch.norm(delayGrad - delay.delay.grad).item() < 1e-4, 'CustomDelayGradient and AutoGrad1 results must match.')

# AutoGrad 2:
# first delay followed by Psp opeartion

# reset previous gradient
delay.delay.grad = None

psp = slayer.psp(spikeIn)
pspDelayed = delay(psp)
uOut = fc1(pspDelayed)
spikeOut = slayer.spike(fc1(slayer.psp(delay(spikeIn))))

# loss
error = snn.loss(netParams).to(device)
loss  = error.spikeTime(spikeOut, spikeDes)

loss.backward()

class TestAutoGrad2(unittest.TestCase):
	def test(self):
		# print('CustomDelayGradient - autoGrad2:', torch.norm(delayGrad - delay.delay.grad).item())
		self.assertTrue(torch.norm(delayGrad - delay.delay.grad).item() < 1e-4, 'CustomDelayGradient and AutoGrad2 results must match.')

def plot():
	plt.figure(1)
	plt.subplot(2, 1, 1)
	plt.plot(slayer.psp(spikeOut - spikeDes)[0].cpu().data.numpy().reshape((Nout, -1)).transpose())
	plt.xlabel('Time bins')
	plt.ylabel('Output layer Error')

	plt.subplot(2, 1, 2)
	plt.plot(deltaRec[0].cpu().data.numpy().reshape((Nout, -1)).transpose())
	plt.xlabel('Time bins')
	plt.ylabel('Output layer Delta')

	plt.figure(2)
	plt.plot(errorRec[0].cpu().data.numpy().reshape((Nin, -1)).transpose())
	plt.xlabel('Time bins')
	plt.ylabel('Hidden layer Error')

	plt.figure(3)
	plt.subplot(2, 1, 1)
	plt.plot(pspDelayed[0].cpu().data.numpy().reshape((Nin, -1)).transpose())
	plt.xlabel('Time bins')
	plt.ylabel('Hidden layer delayed PSP')

	plt.subplot(2, 1, 2)
	plt.plot(pspDelayed[0].cpu().data.numpy().reshape((Nin, -1)).transpose()[:, 0], label='PSP')
	plt.plot(np.cumsum(dpspDelayed_dt[0].cpu().data.numpy().reshape((Nin, -1)).transpose()[:, 0]) * Ts, label='PSP reconstructed')
	plt.xlabel('Time bins')
	plt.ylabel('PSP/PSP reconstructed from time derivative')
	plt.legend()

	plt.show()


if verbose is True:	
	if bool(os.environ.get('DISPLAY', None)):
		plot()

if __name__ == '__main__':	
	unittest.main()