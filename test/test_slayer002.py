import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import numpy as np
import matplotlib.pyplot as plt
from slayer import spikeLayer
from data_reader import SlayerParams
import torch

###############################################################################
# testing against known snn data ##############################################
net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/snnData/network.yaml")

Ns = int(net_params['simulation']['tSample'] / net_params['simulation']['Ts'])
Nin = int(net_params['layer'][0]['dim'])
Nhid = int(net_params['layer'][1]['dim'])
Nout = int(net_params['layer'][2]['dim'])

# initialize slayer
# slayer = spikeLayer(net_params['neuron'], net_params['simulation'])
slayer = spikeLayer(net_params['neuron'], net_params['simulation'], fullRefKernel = True)

# define network functions
spike = slayer.spike()
psp   = slayer.psp()
fc1   = slayer.dense(Nin, Nhid)
fc2   = slayer.dense(Nhid, Nout)

# load input spikes
spikeAER = np.loadtxt('test_files/snnData/spikeIn.txt')
spikeAER[:,0] /= net_params['simulation']['Ts']
spikeAER[:,1] -= 1

spikeData = np.zeros((Nin, Ns))
for (tID, nID) in np.rint(spikeAER).astype(int):
	if tID < Ns : spikeData[nID, tID] = 1/net_params['simulation']['Ts']
spikeIn = torch.FloatTensor(spikeData.reshape((1, Nin, 1, 1, Ns))).to(torch.device('cuda'))

spikeDataGT = np.loadtxt('test_files/snnData/spikeInMat.txt')[:,:-1]
print('Spike Load Error:', np.linalg.norm(spikeDataGT - spikeData))


# load input layer weight
W1 = np.loadtxt('test_files/snnData/w1learned.txt')
fc1.weight = torch.nn.Parameter(torch.FloatTensor(W1.reshape((Nhid, Nin, 1, 1, 1))).to(fc1.weight.device), requires_grad = True)
# print(W1.shape)
# print(W1)
# print(fc1.weight.shape)
# print(fc1.weight.reshape((Nhid, Nin)))
data = np.loadtxt('test_files/snnData/spikeFuncHid.txt')[0:-1]
# data = np.loadtxt('test_files/snnData/spikeFuncHid04.txt')[0:-1]
aIn      = psp(spikeIn)
uHid     = fc1(aIn)
uHidPre  = torch.tensor(uHid, requires_grad=False)

spikeHid = spike(uHid)
s2 = spikeHid.reshape((Nhid, Ns)).cpu().data.numpy()
s2AER = np.argwhere(s2 > 0)
s2GT  = np.loadtxt('test_files/snnData/spikeHid.txt')
s2GT[:,0] /= net_params['simulation']['Ts']
s2GT[:,1] -= 1

spikeHidData = np.zeros((Nhid, Ns))
for (tID, nID) in np.rint(s2GT).astype(int):
	if tID < Ns : spikeHidData[nID, tID] = 1/net_params['simulation']['Ts']
spikeHidGT = torch.FloatTensor(spikeHidData.reshape((1, Nhid, 1, 1, Ns))).to(torch.device('cuda'))

# load hidden layer weight
W2 = np.loadtxt('test_files/snnData/w2learned.txt')
fc2.weight = torch.nn.Parameter(torch.FloatTensor(W2.reshape((Nout, Nhid, 1, 1, 1))).to(fc2.weight.device), requires_grad = True)
dOut = np.loadtxt('test_files/snnData/spikeFuncOut.txt')[0:-1]
uOut = fc2(psp(spikeHid))
# uOut = fc2(psp(spikeHidGT))
uOutPre = torch.tensor(uOut, requires_grad=False)
spikeOut = spike(uOut)
s3 = spikeOut.reshape((Nout, Ns)).cpu().data.numpy()
s3AER = np.argwhere(s3 > 0)
s3GT  = np.loadtxt('test_files/snnData/spikeOut.txt')
s3GT[:,0] /= net_params['simulation']['Ts']
s3GT[:,1] -= 1

plt.figure(1)
plt.subplot(3, 1, 1)
addrEvent = np.argwhere(spikeData > 0)
plt.plot(addrEvent[:, 1], addrEvent[:, 0], 'o', label = 'Ground Truth')
plt.plot(spikeAER [:,0],  spikeAER[:, 1],  '.', label = 'Actual spikes')
# plt.xlabel('Time bins')
plt.ylabel('Neuron ID')
plt.title('Input Spikes')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(s2GT [:, 0], s2GT [:, 1], 'o', label = 'Ground Truth')
plt.plot(s2AER[:, 1], s2AER[:, 0], '.', label = 'Actual spikes')
# plt.xlabel('Time bins')
plt.ylabel('Neuron ID')
plt.title('Hidden Layer Spikes')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(s3GT [:, 0], s3GT [:, 1], 'o', label = 'Ground Truth')
plt.plot(s3AER[:, 1], s3AER[:, 0], '.', label = 'Actual spikes')
plt.xlabel('Time bins')
plt.ylabel('Neuron ID')
plt.title('Hidden Layer Spikes')
plt.legend()

plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(uHid    .reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose(), label = 'u : after refractory')
plt.plot(spikeHid.reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose(), label = 'spike')
plt.plot(uHidPre .reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose(), label = 'u : before refractory')
# plt.xlabel('Time bins')
plt.title('Hidden Layer')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(uOut    .reshape((Nout, Ns)).cpu().data.numpy()[0].transpose(), label = 'u : after refractory')
plt.plot(spikeOut.reshape((Nout, Ns)).cpu().data.numpy()[0].transpose(), label = 'spike')
plt.xlabel('Time bins')
plt.title('Output Layer')
plt.legend()

plt.figure(40)
plt.subplot(2, 1, 1)
plt.plot(uHid    .reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose() - data[:, 1], label = 'u : after refractory')
plt.plot(spikeHid.reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose() - data[:, 2], label = 'spike')
plt.plot(uHidPre .reshape((Nhid, Ns)).cpu().data.numpy()[0].transpose() - data[:, 0], label = 'u : before refractory')
plt.title('Error: Hidden Layer')
plt.legend()
# plt.axis((None, None, -0.1, 0.1))

plt.subplot(2, 1, 2)
plt.plot(uOut    .reshape((Nout, Ns)).cpu().data.numpy()[0].transpose() - dOut[:, 1], label = 'u : after refractory')
plt.plot(spikeOut.reshape((Nout, Ns)).cpu().data.numpy()[0].transpose() - dOut[:, 2], label = 'spike')
plt.plot(uOutPre .reshape((Nout, Ns)).cpu().data.numpy()[0].transpose() - dOut[:, 0], label = 'u : before refractory')
plt.title('Error: Output Layer')
plt.xlabel('Time bins')
plt.legend()
# plt.axis((None, None, -0.1, 0.1))

# u = uHidPre.reshape((Nhid, Ns)).cpu().data.numpy()[4]
u = data.transpose()[0]
print(u.shape)
s = np.zeros(u.shape)
theta = net_params['neuron']['theta']
Ts = net_params['simulation']['Ts']
refKernel = slayer.refKernel.cpu().data.numpy()
print(refKernel.shape)
for t in np.arange(Ns):
	if u[t] >= theta:
		s[t] = 1/Ts
		if t + refKernel.size <= Ns: 
			u[t:t+refKernel.size] += refKernel
		else:
			u[t:] += refKernel[:Ns-t]

plt.figure(100)
plt.subplot(2, 1, 1)
plt.plot(u, 'o', label = 'Calculated')
# plt.plot(uHid.reshape((Nhid, Ns)).cpu().data.numpy()[4].transpose(), label = 'u : after refractory')
plt.plot(data.transpose()[1], label = 'Ground Truth')
plt.ylabel('u')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(s, 'o', label = 'Calculated')
# plt.plot(spikeHid.reshape((Nhid, Ns)).cpu().data.numpy()[4].transpose(), label = 'spike')
plt.plot(data.transpose()[2], label = 'Ground Truth')
plt.legend()
plt.xlabel('Time bins')
plt.ylabel('spike')
			
plt.show()