import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
from slayerParams import yamlParams as SlayerParams
from slayer import spikeLayer
from slayerSNN import loss as spikeLoss
from spikeClassifier import spikeClassifier as predict

###############################################################################
# testing numSpikesError ######################################################
net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/snnData/network.yaml")

Ns   = int(net_params['simulation']['tSample'] / net_params['simulation']['Ts'])
Nin  = int(net_params['layer'][0]['dim'])
Nout = int(net_params['layer'][1]['dim'])
net_params['training']['error']['type'] = 'NumSpikes'

device = torch.device('cuda')
# device = torch.device('cuda:3')

class Network(torch.nn.Module):
	def __init__(self, net_params):
		super(Network, self).__init__()
		# initialize slayer
		slayer = spikeLayer(net_params['neuron'], net_params['simulation'], fullRefKernel=True)

		self.slayer = slayer
		# define network functions
		self.fc1   = slayer.dense(Nin, Nout)
		W1 = np.loadtxt('test_files/snnData/w1Initial.txt')
		self.fc1.weight = torch.nn.Parameter(torch.FloatTensor(W1.reshape((Nout, Nin , 1, 1, 1))).to(self.fc1.weight.device), requires_grad = True)
		
	def forward(self, spikeInput):
		return self.slayer.spike(self.fc1(self.slayer.psp(spikeInput)))
		
snn = Network(net_params).to(device)

# load input spikes
spikeAER = np.loadtxt('test_files/snnData/spikeIn.txt')
spikeAER[:,0] /= net_params['simulation']['Ts']
spikeAER[:,1] -= 1

spikeData = np.zeros((Nin, Ns))
for (tID, nID) in np.rint(spikeAER).astype(int):
	if tID < Ns : spikeData[nID, tID] = 1/net_params['simulation']['Ts']
spikeIn = torch.FloatTensor(spikeData.reshape((1, Nin, 1, 1, Ns))).to(device)

spikeOut = snn.forward(spikeIn)

# desired class must be a boolean tensor whose first four dimension is same as output tensor
# and last dimension (time dimension) is ONE
desiredClass = torch.zeros((1, Nout, 1, 1, 1)).to('cpu')
desiredClass[0,4,0,0,0] = 1 # assuming true class is class 5

# error = spikeLoss(snn.slayer, net_params['training']['error']).to(device)
error = spikeLoss(net_params).to(device)
loss = error.numSpikes(spikeOut, desiredClass)

loss.backward()

# print('Output Class is :', predict.getClass(spikeOut.reshape((1, 1, 1, Nout, Ns))))
# print('Output Class is :', predict.getClass(spikeOut))
# print('Correct Class is:', 21)
class testSpikeClassifier(unittest.TestCase):
	def test(self):
		self.assertEqual(predict.getClass(spikeOut), 21, 'Correct class is 21.')

# output  = spikeOut.reshape((Nout, Ns)).cpu().data.numpy()
# desired =     loss.reshape((Nout, Ns)).cpu().data.numpy()
# outputAER  = np.argwhere(output  > 0)
# desiredAER = np.argwhere(desired > 0)

# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(outputAER [:,1], outputAER [:,0], 'o', label='Output Spikes')

# plt.subplot(2, 1, 2)
# plt.plot(loss.cpu().data.numpy()[0,0,0,0,:])
# # plt.plot(loss.cpu().data.numpy()[0,5,0,0,:])
# plt.show()