import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
from slayerLoihi import spikeLayer
from quantizeParams import quantizeWeights
from nmnistTrain import *

netParams = snn.params('network.yaml')

# Define the cuda device to run the code on.
device = torch.device('cuda')

# Create network instance.
net = Network(netParams).to(device)

# load saved net
net.load_state_dict(torch.load('Trained/nmnistNet.pt'))

testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'], 
						    sampleFile  =netParams['training']['path']['test'],
						    samplingTime=netParams['simulation']['Ts'],
						    sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

# generate Loihi parameters
stats = learningStats()

for i, (input, target, label) in enumerate(testLoader, 0):
	input  = input.to(device)
	target = target.to(device) 
	
	output = net.forward(input)

	stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
	stats.testing.numSamples     += len(label)

	# loss = error.numSpikes(output, target)
	# stats.testing.lossSum += loss.cpu().data.item()
	if i%10 == 0:	stats.print(0, i)

genLoihiParams(net)

plt.show()
