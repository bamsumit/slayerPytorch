import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
from slayerSNN import loihi as spikeLayer
from slayerSNN import params as SlayerParams
import torch

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

device = torch.device('cuda')

# load parameters from yaml file
net_params = SlayerParams('test_files/Loihi/network.yaml')
	
if verbose is True:
	print('Neuron Type:', 'LOIHI' )
	print('vThMant :', 80)
	print('vDecay  :', 128 )
	print('iDecay  :', 1024 )
	print('refDelay:', 1 )

# Network structure
Ns   = int(net_params['simulation']['tSample'] / net_params['simulation']['Ts'])
Nin  = int(net_params['layer'][0]['dim'])
Nhid = int(net_params['layer'][1]['dim'])
Nout = int(net_params['layer'][2]['dim'])

slayer = spikeLayer(net_params['neuron'], net_params['simulation']).to(device)

if verbose is True: print('Neuron Threshold =', slayer.neuron['theta'])

# define network functions
fc1   = slayer.dense(Nin, Nhid).to(device)
fc2   = slayer.dense(Nhid, Nout).to(device)

# load input spikes
spikeAER = np.loadtxt('test_files/Loihi/snnData/spikeIn.txt')
spikeAER[:,0] /= net_params['simulation']['Ts']
spikeAER[:,1] -= 1
# spikeAER = np.fliplr(np.loadtxt('loihiInputSpikes.txt'))
spikeAER[:,0] += 49	# 49 not 50 because the input spikes seem to be transmitted without delay

spikeData = np.zeros((Nin, Ns))
for (tID, nID) in np.rint(spikeAER).astype(int):
	if tID < Ns : spikeData[nID, tID] = 1/net_params['simulation']['Ts']
# spikeIn = torch.IntTensor(spikeData.reshape((1, Nin, 1, 1, Ns))).to(device)
spikeIn = torch.FloatTensor(spikeData.reshape((1, Nin, 1, 1, Ns))).to(device)

# load hidden spikes
spikeAEH = np.loadtxt('test_files/Loihi/LoihiData/loihiHiddenSpikes.txt')
spikeData = np.zeros((Nhid, Ns))
for (nID, tID) in np.rint(spikeAEH).astype(int):
	if tID < Ns : spikeData[nID, tID] = 1 #/net_params['simulation']['Ts']
# spikeHid = torch.FloatTensor(spikeData.reshape((1, Nhid, 1, 1, Ns))).to(device)
spikeHid = torch.IntTensor(spikeData.reshape((1, Nhid, 1, 1, Ns))).to(device)

# load weights
wScale = 4
W1 = np.round(np.loadtxt('test_files/Loihi/snnData/w1learned.txt') * wScale / 2 )*2
W2 = np.round(np.loadtxt('test_files/Loihi/snnData/w2learned.txt') * wScale / 2 )*2
fc1.weight = torch.nn.Parameter(torch.FloatTensor(W1.reshape((Nhid, Nin , 1, 1, 1))).to(fc1.weight.device), requires_grad = True)
fc2.weight = torch.nn.Parameter(torch.FloatTensor(W2.reshape((Nout, Nhid, 1, 1, 1))).to(fc2.weight.device), requires_grad = True)

# run the network
# wSpikeIn = torch.IntTensor( np.dot( W1.astype(int), spikeIn.cpu().data.numpy().reshape((-1, Ns)) )
# 						   ).to(device).reshape((1, -1, 1, 1, Ns))
wSpikeIn = fc1(spikeIn)
# spikeHid, uHid = slayer.spikeSVI(wSpikeIn)[0:2]
spikeHid, uHid = slayer.spikeLoihiFull(wSpikeIn)[0:2]
# spikeHid[...,1:Ns] = spikeHid[...,0:Ns-1]	# shift spike by one to simulate axonal delay of 1
spikeHid = slayer.delayShift(spikeHid, 1)	# shift spike by one to simulate axonal delay of 1

# wSpikeHid = torch.IntTensor( np.dot( W2.astype(int), spikeHid.cpu().data.numpy().reshape((-1, Ns)) )
# 						   ).to(device).reshape((1, -1, 1, 1, Ns))
wSpikeHid = fc2(spikeHid)
# spikeOut, uOut = slayer.spikeSVI(wSpikeHid)[0:2]
spikeOut, uOut = slayer.spikeLoihiFull(wSpikeHid)[0:2]
# spikeOut[...,1:Ns] = spikeOut[...,0:Ns-1]	# shift spike by one to simulate axonal delay of 1
spikeOut = slayer.delayShift(spikeOut, 1)	# shift spike by one to simulate axonal delay of 1

if verbose is True: print(fc2.weight.flatten())

loihiSpikes = np.loadtxt('test_files/Loihi/LoihiData/loihiInputSpikes.txt')
hiddenAER = np.argwhere(spikeHid.reshape((Nhid, Ns)).cpu().data.numpy() > 0)
hiddenGT  = np.loadtxt('test_files/Loihi/LoihiData/loihiHiddenSpikes.txt')
outputAER = np.argwhere(spikeOut.reshape((Nout, Ns)).cpu().data.numpy() > 0)
outputGT  = np.loadtxt('test_files/Loihi/LoihiData/loihiOutputSpikes.txt')
uGTHid = np.loadtxt('test_files/Loihi/LoihiData/loihiHiddenVoltage24.txt')
uGTOut = np.loadtxt('test_files/Loihi/LoihiData/loihiOutputVoltage.txt')

class testLoihiSpikes(unittest.TestCase):
	def testHidden(self):
		spike   = torch.zeros((Nhid, Ns))
		spikeGT = torch.zeros((Nhid, Ns))
		
		for (nID, tID) in np.rint(hiddenAER).astype(int):
			spike[nID, tID] = 1/net_params['simulation']['Ts']
		
		for (nID, tID) in np.rint(hiddenGT).astype(int):
			spikeGT[nID, tID] = 1/net_params['simulation']['Ts']	
			
		error = torch.norm(spike - spikeGT).item()
		
		self.assertTrue(error<1e-3, 'Hidden spike and ground truth must match.')

	def testOutput(self):
		spike   = torch.zeros((Nout, Ns))
		spikeGT = torch.zeros((Nout, Ns))
		
		for (nID, tID) in np.rint(outputAER).astype(int):
			spike[nID, tID] = 1/net_params['simulation']['Ts']
		
		for (nID, tID) in np.rint(outputGT).astype(int):
			spikeGT[nID, tID] = 1/net_params['simulation']['Ts']	
			
		error = torch.norm(spike - spikeGT).item()
		
		self.assertTrue(error<1e-3, 'Output spike and ground truth must match.')


class testLoihiVoltage(unittest.TestCase):
	def testHidden(self):
		error = torch.norm(uHid[0,24,0,0,:-1] - torch.FloatTensor(uGTHid).to(device)).item()
		self.assertTrue(error<1e-3, 'Hidden voltage and ground truth must match.')

	def testOutput(self):
		error = torch.norm(uOut[...,:-1] - torch.FloatTensor(uGTOut).to(device)).item()
		self.assertTrue(error<1e-3, 'Hidden voltage and ground truth must match.')


if verbose is True:
	if bool(os.environ.get('DISPLAY', None)):
		plt.figure(1)
		ax1 = plt.subplot(3, 1, 1)
		plt.plot(loihiSpikes [:,1],  loihiSpikes[:, 0],  'o', label='Loihi')
		plt.plot(spikeAER [:,0],  spikeAER[:, 1],  '.', label='slayerLoihiInt')
		plt.ylabel('Neuron ID')
		plt.title('Input Spikes')
		plt.legend()

		ax2 = plt.subplot(3, 1, 2)
		plt.plot(hiddenGT [:, 1], hiddenGT [:, 0], 'o', label='Loihi')
		plt.plot(hiddenAER[:, 1], hiddenAER[:, 0], '.', label='slayerLoihiInt')
		ax2.set_xlim(ax1.get_xlim())
		plt.ylabel('Neuron ID')
		plt.title('Hidden Layer Spikes')
		plt.legend()

		ax3 = plt.subplot(3, 1, 3)
		plt.plot(outputGT [:, 1], outputGT [:, 0], 'o', label='Loihi')
		plt.plot(outputAER[:, 1], outputAER[:, 0], '.', label='slayerLoihiInt')
		ax3.set_xlim(ax1.get_xlim())
		plt.xlabel('Time bins')
		plt.ylabel('Neuron ID')
		plt.title('Output Layer Spikes')
		plt.legend()

		plt.figure(2)
		plt.subplot(2, 1, 1)
		uGT = np.loadtxt('test_files/Loihi/LoihiData/loihiHiddenVoltage24.txt')	
		plt.plot(uGT.flatten(), label='Loihi')
		plt.plot(uHid[0,24,0,0,:].cpu().data.numpy().flatten(), label='slayerLoihiInt')
		plt.ylabel('Membrane Potential')
		plt.title('Hidden Layer')
		plt.legend()

		plt.subplot(2, 1, 2)
		plt.plot(uHid[0,24,0,0,:].cpu().data.numpy().flatten()[:-1] - uGT.flatten())
		plt.ylabel('Membrane Potential Error')
		plt.xlabel('Time bins')

		plt.figure(3)
		plt.subplot(2, 1, 1)
		uGT = np.loadtxt('test_files/Loihi/LoihiData/loihiOutputVoltage.txt')
		plt.plot(uGT.flatten(), label='Loihi')
		plt.plot(uOut.cpu().data.numpy().flatten(), label='slayerLoihiInt')
		plt.ylabel('Membrane Potential')
		plt.title('Output Layer')
		plt.legend()

		plt.subplot(2, 1, 2)
		plt.plot(uOut.cpu().data.numpy().flatten()[:-1] - uGT.flatten())
		plt.ylabel('Membrane Potential Error')
		plt.xlabel('Time bins')

		# print(uOut.cpu().data.numpy().flatten()[:10])
		# print(uHid[0,0,0,0,:].cpu().data.numpy().flatten()[:10])
		# print(uGT.flatten()[:10])
		
		plt.show()
