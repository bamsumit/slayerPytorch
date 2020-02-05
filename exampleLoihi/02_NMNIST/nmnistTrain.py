import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import zipfile

# Read SNN configuration from yaml file
netParams = snn.params('network.yaml')

# Ts   = netParams['simulation']['Ts']
# Ns   = int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
# Nin  = int(netParams['layer'][0]['dim'])
# Nhid = int(netParams['layer'][1]['dim'])
# Nout = int(netParams['layer'][2]['dim'])

# Extract NMNISTsmall dataset
with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
    for member in zip_file.namelist():
        if not os.path.exists('./' + member):
            zip_file.extract(member, './')
			
# Define dataset module
class nmnistDataset(Dataset):
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
                        ).toSpikeTensor(torch.zeros((2,34,34,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel,...] = 1
        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel
    
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
        self.fc1   = slayer.dense((34*34*2), 512)
        self.fc2   = slayer.dense(512, 10)

    def forward(self, spikeInput):
        spike = self.slayer.spikeLoihi(self.fc1(spikeInput))
        spike = self.slayer.delayShift(spike, 1)
        # A minimum axonal delay of 1 tick is required in Loihi hardare
        spike = self.slayer.spikeLoihi(self.fc2(spike))
        spike = self.slayer.delayShift(spike, 1)
        return spike

# Define Loihi parameter generator
def genLoihiParams(net):
	fc1Weights = snn.uitls.quantize(net.fc1.weight, 2).flatten().cpu().data.numpy()
	fc2Weights = snn.uitls.quantize(net.fc2.weight, 2).flatten().cpu().data.numpy()

	np.save('Trained/NMNISTFc1.npy', fc1Weights)
	np.save('Trained/NMNISTFc2.npy', fc2Weights)

	plt.figure(11)
	plt.hist(fc1Weights, 256)
	plt.title('fc1 weights')

	plt.figure(12)
	plt.hist(fc2Weights, 256)
	plt.title('fc2 weights')

if __name__ == '__main__':
	# Define the cuda device to run the code on.
	device = torch.device('cuda')

	# Create network instance.
	net = Network(netParams).to(device)

	# Create snn loss instance.
	error = snn.loss(netParams, snn.loihi).to(device)

	# Define optimizer module.
	# optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
	optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)

	# Dataset and dataLoader instances.
	trainingSet = nmnistDataset(datasetPath =netParams['training']['path']['in'], 
								sampleFile  =netParams['training']['path']['train'],
								samplingTime=netParams['simulation']['Ts'],
								sampleLength=netParams['simulation']['tSample'])
	trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

	testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'], 
								sampleFile  =netParams['training']['path']['test'],
								samplingTime=netParams['simulation']['Ts'],
								sampleLength=netParams['simulation']['tSample'])
	testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

	# Learning stats instance.
	stats = snn.utils.stats()
	
	# Visualize the input spikes (first five samples).
	for i in range(5):
		input, target, label = trainingSet[i]
		snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

	for epoch in range(100):
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
			
			input  = input.to(device)
			target = target.to(device) 

			output = net.forward(input)

			stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
			stats.testing.numSamples     += len(label)

			loss = error.numSpikes(output, target)
			stats.testing.lossSum += loss.cpu().data.item()
			stats.print(epoch, i)

		# Update testing stats.
		stats.update()
		stats.plot(saveFig=True, path='Trained/')
		if stats.training.bestLoss is True:	torch.save(net.state_dict(), 'Trained/nmnistNet.pt')

	# Save training data
	stats.save('Trained/')
	net.load_state_dict(torch.load('Trained/nmnistNet.pt'))
	genLoihiParams(net)

	# Plot the results
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
