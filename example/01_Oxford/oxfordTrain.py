###############################################################################
# This is an example for training to produce Oxford spikes. The task is to 
# train a multilayer SNN to produce spike raster that resembles Oxford house.
# The input and output both consists of 200 neurons each and the spkes span 
# approximately 1900ms. The input and output spike pair are taken from 
# SuperSpike repository (https://github.com/fzenke/pub2018superspike).
###############################################################################

import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../../src")

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn
from learningStats import learningStats

###############################################################################
# oxford spike learning ######################################################
netParams = snn.params("oxford/oxford.yaml")

Ts   = netParams['simulation']['Ts']
Ns   = int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
Nin  = int(netParams['layer'][0]['dim'])
Nhid = int(netParams['layer'][1]['dim'])
Nout = int(netParams['layer'][2]['dim'])

device = torch.device('cuda:0')

class Network(torch.nn.Module):
	def __init__(self, netParams):
		super(Network, self).__init__()
		# initialize slayer
		slayer = snn.layer(netParams['neuron'], netParams['simulation'])

		self.slayer = slayer
		# define network functions
		self.fc1   = slayer.dense(Nin, Nhid)
		self.fc2   = slayer.dense(Nhid, Nout)
		# self.fc1.weight = torch.nn.Parameter(100 * self.fc1.weight)
		# self.fc2.weight = torch.nn.Parameter(100 * self.fc2.weight)

	def forward(self, spikeInput):
		spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(spikeInput)))
		spikeLayer2 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1)))
		return spikeLayer2

net = Network(netParams).to(device)

# define error module
error = snn.loss(netParams).to(device)

# define optimizer module
# optimizer = torch.optim.SGD(snn.parameters(), lr = 0.001)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

inTD  = snn.io.read1Dspikes('oxford/input.bs1')
# inTD.p -= 1
# print(inTD.p.min())
input = inTD.toSpikeTensor(torch.zeros((1, 1, Nin, Ns)), 
			samplingTime=Ts).reshape((1, Nin, 1, 1, Ns)).to(device)

desTD   = snn.io.read1Dspikes('oxford/output.bs1')
# desTD.p -= 1
desired = desTD.toSpikeTensor(torch.zeros((1, 1, Nout, Ns)), 
			samplingTime=Ts).reshape((1, Nout, 1, 1, Ns)).to(device)

# print(inTD.toSpikeArray().shape)
# print(input.shape)


# snn.io.showTD(inTD)
# snn.io.showTD(desTD)
snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((1, Nin, -1)).cpu().data.numpy()))
snn.io.showTD(snn.io.spikeArrayToEvent(desired.reshape((1, Nout, -1)).cpu().data.numpy()))

losslog = list()

stats = learningStats()

for epoch in range(1000):
	output = net.forward(input)
	
	loss = error.spikeTime(output, desired)

	stats.training.numSamples = 1
	stats.training.lossSum = loss.cpu().data.item()
	
	if epoch%10 == 0:	stats.print(epoch)
	losslog.append(loss.cpu().data.numpy())
	
	# if epoch==0:
	# 	minLoss = loss
	# 	bestNet = copy.deepcopy(net)
	# else:
	# 	if loss < minLoss:
	# 		minLoss = loss
	# 		bestNet = copy.deepcopy(net)
	stats.training.update()
	if stats.training.bestLoss is True:	bestNet = copy.deepcopy(net)

	if loss < 1e-5:	break

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# if epoch%100 == 0:
	# 	snn.io.showTD(snn.io.spikeArrayToEvent(output.reshape((1, -1, Ns)).cpu().data.numpy()))


output = bestNet.forward(input)
loss = error.spikeTime(output, desired)
print(loss.item())
snn.io.showTD(snn.io.spikeArrayToEvent(output.reshape((1, -1, Ns)).cpu().data.numpy()))

plt.figure(1)
plt.semilogy(losslog)
plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure(2)
desAER = np.argwhere(desired.reshape((Nout, Ns)).cpu().data.numpy() > 0)
outAER = np.argwhere(output.reshape((Nout, Ns)).cpu().data.numpy() > 0)
plt.plot(desAER[:, 1], desAER[:, 0], 'o', label='desired')
plt.plot(outAER[:, 1], outAER[:, 0], '.', label='actual')
plt.title('Spike Raster')
plt.xlabel('time')
plt.ylabel('neuron ID')
plt.legend()

plt.show()

