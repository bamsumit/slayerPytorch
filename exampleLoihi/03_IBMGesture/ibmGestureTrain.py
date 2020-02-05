import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn

# Define dataset module
class IBMGestureDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        # Read inoput and label
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        # Read input spike
        inputSpikes = snn.io.read2Dspikes(
                        self.path + str(inputIndex.item()) + '.bs2'
                        ).toSpikeTensor(torch.zeros((2,128,128,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((11, 1, 1, 1))
        desiredClass[classLabel,...] = 1
        
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]
		
# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 16, 5, padding=2, weightScale=10)
        self.conv2 = slayer.conv(16, 32, 3, padding=1, weightScale=50)
        self.pool1 = slayer.pool(4)
        self.pool2 = slayer.pool(2)
        self.pool3 = slayer.pool(2)
        self.fc1   = slayer.dense((8*8*32), 512)
        self.fc2   = slayer.dense(512, 11)
        self.drop  = slayer.dropout(0.1)

    def forward(self, spikeInput):
        spike = self.slayer.spikeLoihi(self.pool1(spikeInput )) # 32, 32, 2
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv1(spike)) # 32, 32, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool2(spike)) # 16, 16, 16
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv2(spike)) # 16, 16, 32
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.pool3(spike)) #  8,  8, 32
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
        spike = self.slayer.delayShift(spike, 1)
        
        spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 11
        spike = self.slayer.delayShift(spike, 1)
        
        return spike
		
# Define Loihi parameter generator
def genLoihiParams(net):
	fc1Weights   = snn.uitls.quantize(net.fc1.weight  , 2).flatten().cpu().data.numpy()
	fc2Weights   = snn.uitls.quantize(net.fc2.weight  , 2).flatten().cpu().data.numpy()
	conv1Weights = snn.uitls.quantize(net.conv1.weight, 2).flatten().cpu().data.numpy()
	conv2Weights = snn.uitls.quantize(net.conv2.weight, 2).flatten().cpu().data.numpy()
	pool1Weights = snn.uitls.quantize(net.pool1.weight, 2).flatten().cpu().data.numpy()
	pool2Weights = snn.uitls.quantize(net.pool2.weight, 2).flatten().cpu().data.numpy()
	pool3Weights = snn.uitls.quantize(net.pool3.weight, 2).flatten().cpu().data.numpy()

	np.save('Trained/fc1.npy'  , fc1Weights)
	np.save('Trained/fc2.npy'  , fc2Weights)
	np.save('Trained/conv1.npy', conv1Weights)
	np.save('Trained/conv2.npy', conv2Weights)
	np.save('Trained/pool1.npy', pool1Weights)
	np.save('Trained/pool2.npy', pool2Weights)
	np.save('Trained/pool3.npy', pool3Weights)

	plt.figure(11)
	plt.hist(fc1Weights  , 256)
	plt.title('fc1 weights')

	plt.figure(12)
	plt.hist(fc2Weights  , 256)
	plt.title('fc2 weights')

	plt.figure(13)
	plt.hist(conv1Weights, 256)
	plt.title('conv1 weights')

	plt.figure(14)
	plt.hist(conv2Weights, 256)
	plt.title('conv2 weights')

	plt.figure(15)
	plt.hist(pool1Weights, 256)
	plt.title('pool1 weights')

	plt.figure(16)
	plt.hist(pool2Weights, 256)
	plt.title('pool2 weights')

	plt.figure(17)
	plt.hist(pool3Weights, 256)
	plt.title('pool3 weights')
	
if __name__ == '__main__':
	netParams = snn.params('network.yaml')
	
	# Define the cuda device to run the code on.
	device = torch.device('cuda')
	# deviceIds = [2, 3]

	# Create network instance.
	net = Network(netParams).to(device)
	# net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

	# Create snn loss instance.
	error = snn.loss(netParams, snn.loihi).to(device)

	# Define optimizer module.
	# optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
	optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)

	# Dataset and dataLoader instances.
	trainingSet = IBMGestureDataset(datasetPath =netParams['training']['path']['in'], 
									sampleFile  =netParams['training']['path']['train'],
									samplingTime=netParams['simulation']['Ts'],
									sampleLength=netParams['simulation']['tSample'])
	trainLoader = DataLoader(dataset=trainingSet, batch_size=4, shuffle=True, num_workers=1)

	testingSet = IBMGestureDataset(datasetPath  =netParams['training']['path']['in'], 
								   sampleFile  =netParams['training']['path']['test'],
								   samplingTime=netParams['simulation']['Ts'],
								   sampleLength=netParams['simulation']['tSample'])
	testLoader = DataLoader(dataset=testingSet, batch_size=4, shuffle=True, num_workers=1)

	# Learning stats instance.
	stats = snn.utils.stats()
	
	# Visualize the input spikes (first five samples).
	for i in range(5):
		input, target, label = trainingSet[i]
		snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 128, 128, -1)).cpu().data.numpy()))
		
	# for epoch in range(500):
	for epoch in range(5):
		tSt = datetime.now()

		# Training loop.
		for i, (input, target, label) in enumerate(trainLoader, 0):
			net.train()

			# Move the input and target to correct GPU.
			input  = input.to(device)
			target = target.to(device) 

			# Forward pass of the network.
			output = net.forward(input)

			# Gather the training stats.
			stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
			stats.training.numSamples     += len(label)

			# Calculate loss.
			loss = error.numSpikes(output, target)

			# Reset gradients to zero.
			optimizer.zero_grad()

			# Backward pass of the network.
			loss.backward()

			# Update weights.
			optimizer.step()

			# Gather training loss stats.
			stats.training.lossSum += loss.cpu().data.item()

			# Display training stats.
			stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

		# Testing loop.
		# Same steps as Training loops except loss backpropagation and weight update.
		for i, (input, target, label) in enumerate(testLoader, 0):
			net.eval()
			with torch.no_grad():
				input  = input.to(device)
				target = target.to(device) 

			output = net.forward(input)

			stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
			stats.testing.numSamples     += len(label)

			loss = error.numSpikes(output, target)
			stats.testing.lossSum += loss.cpu().data.item()
			stats.print(epoch, i)

		# Update stats.
		stats.update()
		stats.plot(saveFig=True, path='Trained/')
		if stats.training.bestLoss is True: torch.save(net.state_dict(), 'Trained/ibmGestureNet.pt')

	# Save training data
	stats.save('Trained/')
	net.load_state_dict(torch.load('Trained/ibmGestureNet.pt'))
	genLoihiParams(net)

	# Plot the results.
	# Learning loss
	plt.figure(1)
	plt.semilogy(stats.training.lossLog, label='Training')
	plt.semilogy(stats.testing .lossLog, label='Testing')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	# Learning accuracy
	plt.figure(2)
	plt.plot(stats.training.accuracyLog, label='Training')
	plt.plot(stats.testing .accuracyLog, label='Testing')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.show()
