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

device = torch.device('cuda')
netParams = snn.params('test_files/nmnistNet.yaml')

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
	def __init__(self, netParams, device=device):
		super(Network, self).__init__()
		# initialize slayer
		slayer = snn.layer(netParams['neuron'], netParams['simulation'], device=device)
		
		self.slayer = slayer
		# define network functions
		self.spike = slayer.spike()
		self.psp   = slayer.psp()
		self.fc1   = slayer.dense((34, 34, 2), 512)
		self.fc1   = slayer.dense((34*34*2), 512)
		self.fc2   = slayer.dense(512, 10)

	def forward(self, spikeInput):
		spikeLayer1 = self.spike(self.fc1(self.psp(spikeInput)))
		spikeLayer2 = self.spike(self.fc2(self.psp(spikeLayer1)))
		
		return spikeLayer2
		# return spikeInput, spikeLayer1, spikeLayer2

# network
net = Network(netParams)

# dataLoader
trainingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['train'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=2)

testingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['test'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=2)

# cost function
error = snn.loss(net.slayer, netParams['training']['error'])

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

# printing functions
printEpoch         = lambda epoch, timeElapsed: print('Epoch: {:4d} \t ({} sec elapsed)'.format(epoch, timeElapsed))
printTrainingStats = lambda cost, accuracy: print('Training: loss = %-12.5g  accuracy = %-6.5g'%(cost, accuracy))
printTestingStats  = lambda cost, accuracy: print('Testing : loss = %-12.5g  accuracy = %-6.5g'%(cost, accuracy))


# training loop
for epoch in range(20):
	epochLoss = 0
	correctSamples = 0
	numSamples = 0
	tSt = datetime.now()
	
	for i, (input, target, label) in enumerate(trainLoader, 0):
		input  = input.to(device)
		target = target.to(device) 
		
		output = net.forward(input)
		
		correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
		numSamples += len(label)

		loss = error.numSpikes(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epochLoss += loss.cpu().data.item()
	
	printEpoch(epoch, (datetime.now() - tSt).total_seconds())
	printTrainingStats(epochLoss/numSamples, correctSamples/numSamples)
	
	correctSamples = 0
	numSamples = 0
	epochLoss = 0
	for i, (input, target, label) in enumerate(testLoader, 0):
		input  = input.to(device)
		target = target.to(device) 
		
		output = net.forward(input)

		correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
		numSamples += len(label)

		loss = error.numSpikes(output, target)
		epochLoss += loss.cpu().data.item()
	
	printTestingStats(epochLoss/numSamples, correctSamples/numSamples)
	
