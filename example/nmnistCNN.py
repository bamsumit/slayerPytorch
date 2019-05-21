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
# device = torch.device('cuda:3')
# deviceIds = [0, 3]
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
		return inputSpikes, desiredClass, classLabel


	def __len__(self):
		return self.samples.shape[0]

# Network definition
class Network(torch.nn.Module):
	# timelog = []

	def __init__(self, netParams):
		super(Network, self).__init__()
		# initialize slayer
		slayer = snn.layer(netParams['neuron'], netParams['simulation'])
		
		self.slayer = slayer
		# define network functions
		self.conv1 = slayer.conv(2, 16, 5, padding=1)
		self.conv2 = slayer.conv(16, 32, 3, padding=1)
		self.conv3 = slayer.conv(32, 64, 3, padding=1)
		self.pool1 = slayer.pool(2)
		self.pool2 = slayer.pool(2)
		self.fc1   = slayer.dense((8, 8, 64), 10)

	def forward(self, spikeInput):
		spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput ))) # 32, 32, 16
		spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1))) # 16, 16, 16
		spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2))) # 16, 16, 32
		spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3))) #  8,  8, 32
		spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4))) #  8,  8, 64
		spikeOut    = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer5))) #  10

		return spikeOut

# network
net = Network(netParams).to(device)
# net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

# dataLoader
trainingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['train'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

testingSet = nmnistDataset(datasetPath=netParams['training']['path']['in'], 
						    sampleFile=netParams['training']['path']['test'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

# cost function
error = snn.loss(netParams).to(device)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

# learning stats
stats = learningStats()

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
		stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

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
		stats.print(epoch, i)
	
	stats.testing.update()

