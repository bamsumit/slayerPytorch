import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# # from data_reader_new import DataReader, SlayerParams
# from data_reader import SlayerParams
# from slayer import spikeLayer
# from spikeLoss import spikeLoss
# from spikeClassifier import spikeClassifier as predict
# # import unittest
# # from txtsaver import txtsaver

import slayerSNN as snn
from learningStats import learningStats

device = torch.device('cuda')
netParams = snn.params('network.yaml')

# Dataloader definition
class nmnistDataset(Dataset):
	def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
		self.path = datasetPath 
		self.samples = np.loadtxt(sampleFile).astype('int')
		self.samplingTime = samplingTime
		self.nTimeBins    = int(sampleLength / samplingTime)

	def __getitem__(self, index):
		inputIndex  = self.samples[index, 0]
		classLabel  = self.samples[index, 1]
		
		inputSpikes = snn.io.read2Dspikes(
						self.path + str(inputIndex.item()) + '.bs2'
						).toSpikeTensor(torch.zeros((2,34,34,self.nTimeBins)),
						samplingTime=self.samplingTime)
		desiredClass = torch.zeros((10, 1, 1, 1))
		desiredClass[classLabel,...] = 1
		# return inputSpikes, desiredClass, classLabel
		return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel


	def __len__(self):
		return self.samples.shape[0]

# Network definition
class Network(torch.nn.Module):
	def __init__(self, netParams):
		super(Network, self).__init__()
		# initialize slayer
		slayer = snn.layer(netParams['neuron'], netParams['simulation'])
		
		self.slayer = slayer
		# define network functions
		# self.fc1   = slayer.dense((34, 34, 2), 512)
		self.fc1   = slayer.dense((34*34*2), 512)
		self.fc2   = slayer.dense(512, 10)

	def forward(self, spikeInput):
		# spikeLayer1 = self.slayer.spike(self.fc1(self.slayer.psp(spikeInput)))
		# spikeLayer2 = self.slayer.spike(self.fc2(self.slayer.psp(spikeLayer1)))

		spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
		spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))		
		
		return spikeLayer2
		# return spikeInput, spikeLayer1, spikeLayer2

# network
net = Network(netParams).to(device)

# dataLoader
trainingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['train'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['test'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=4)

# cost function
# error = snn.loss(net.slayer, netParams['training']['error']).to(device)
error = snn.loss(netParams).to(device)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

stats = learningStats()

# visualize the input spikes (First five samples)
for i in range(5):
	input, target, label = trainingSet[i]
	snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))
	
# training loop
for epoch in range(100):
	stats.training.reset()
	tSt = datetime.now()
	
	for i, (input, target, label) in enumerate(trainLoader, 0):
		input  = input.to(device)
		target = target.to(device) 
		
		output = net.forward(input)
		
		stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
		stats.training.numSamples     += len(label)

		loss = error.numSpikes(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		stats.training.lossSum += loss.cpu().data.item()

		if i%10 == 0:	stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
	
	stats.training.update()
	stats.testing.reset()
	
	for i, (input, target, label) in enumerate(testLoader, 0):
		input  = input.to(device)
		target = target.to(device) 
		
		output = net.forward(input)

		stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
		stats.testing.numSamples     += len(label)

		loss = error.numSpikes(output, target)
		stats.testing.lossSum += loss.cpu().data.item()
		if i%10 == 0:	stats.print(epoch, i)
	
	stats.testing.update()
