###############################################################################
# This is an example for training to produce Oxford spikes. The task is to
# train a multilayer SNN to produce spike raster that resembles Oxford house.
# The input and output both consists of 200 neurons each and the spkes span
# approximately 1900ms. The input and output spike pair are taken from
# SuperSpike repository (https://github.com/fzenke/pub2018superspike).
###############################################################################
import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn

# Read SNN configuration from yaml file
netParams = snn.params('oxford.yaml')

Ts   = netParams['simulation']['Ts']
Ns   = int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
Nin  = int(netParams['layer'][0]['dim'])
Nhid = int(netParams['layer'][1]['dim'])
Nout = int(netParams['layer'][2]['dim'])

# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.fc1   = slayer.dense(Nin, Nhid)
        self.fc2   = slayer.dense(Nhid, Nout)
        
    def forward(self, spikeInput):
        spike = self.slayer.spikeLoihi(self.fc1(spikeInput))
        spike = self.slayer.delayShift(spike, 1)
        # A minimum axonal delay of 1 tick is required in Loihi hardare
        spike = self.slayer.spikeLoihi(self.fc2(spike))
        spike = self.slayer.delayShift(spike, 1)
        return spike

# Define Loihi parameter generator
def genLoihiParams(net):
    fc1Weights = snn.utils.quantize(net.fc1.weight, 2).flatten().cpu().data.numpy()
    fc2Weights = snn.utils.quantize(net.fc2.weight, 2).flatten().cpu().data.numpy()

    np.save('Trained/OxfordFc1.npy', fc1Weights)
    np.save('Trained/OxfordFc2.npy', fc2Weights)

    plt.figure(11)
    plt.hist(fc1Weights, 256)
    plt.title('fc1 weights')

    plt.figure(12)
    plt.hist(fc2Weights, 256)
    plt.title('fc2 weights')

if __name__ == '__main__':
	# define the cuda device to run the code on
	device = torch.device('cuda')

	# create a network instance
	net = Network(netParams).to(device)

	# create snn loss instance
	error = snn.loss(netParams, snn.loihi).to(device)

	# define optimizer module
	# optimizer = torch.optim.SGD(snn.parameters(), lr = 0.001)
	# optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
	optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)

	# Read input spikes and load it to torch tensor
	inTD  = snn.io.read1Dspikes('Spikes/input.bs1')
	input = inTD.toSpikeTensor(torch.zeros((1, 1, Nin, Ns)), samplingTime=Ts).reshape((1, Nin, 1, 1, Ns)).to(device)

	# Read desired spikes and load it to torch tensor
	desTD   = snn.io.read1Dspikes('Spikes/output.bs1')
	desired = desTD.toSpikeTensor(torch.zeros((1, 1, Nout, Ns)), samplingTime=Ts).reshape((1, Nout, 1, 1, Ns)).to(device)

	# Visualize the spike data
	snn.io.showTD(snn.io.spikeArrayToEvent(  input.reshape((1, Nin , -1)).cpu().data.numpy()))
	snn.io.showTD(snn.io.spikeArrayToEvent(desired.reshape((1, Nout, -1)).cpu().data.numpy()))

	# Run the network
	stats = snn.utils.stats()

	for epoch in range(10000):
	    output = net.forward(input)

	    loss = error.spikeTime(output, desired)
	    
	    stats.training.numSamples = 1
	    stats.training.lossSum = loss.cpu().data.item()

	    if epoch%5 == 0: stats.print(epoch)
	        
	    stats.training.update()
	    
	    if stats.training.bestLoss is True: 
	        torch.save(net.state_dict(), 'Trained/oxfordNet.pt')
	    
	    if loss < 1e-5: break

	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()

	# Inference using the best network
	net.load_state_dict(torch.load('Trained/oxfordNet.pt'))
	output = net.forward(input)

	# Save training data
	genLoihiParams(net)

	inpAER = np.argwhere(input.reshape((Nin, Ns)).cpu().data.numpy() > 0)
	desAER = np.argwhere(desired.reshape((Nout, Ns)).cpu().data.numpy() > 0)
	outAER = np.argwhere(output.reshape((Nout, Ns)).cpu().data.numpy() > 0)

	np.savetxt('Trained/OxfordInp.txt', inpAER, fmt='%g')
	np.savetxt('Trained/OxfordOut.txt', outAER, fmt='%g')
	np.savetxt('Trained/OxfordDes.txt', desAER, fmt='%g')

	with open('Trained/loss.txt', 'wt') as loss:
	    loss.write('#%11s\r\n'%('Train'))
	    for i in range(len(stats.training.lossLog)):
	        loss.write('%12.6g\r\n'%(stats.training.lossLog[i]))

	# Plot the results
	plt.figure(1)
	plt.semilogy(stats.training.lossLog)
	plt.title('Training Loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')

	plt.figure(2)
	plt.plot(desAER[:, 1], desAER[:, 0], 'o', label='desired')
	plt.plot(outAER[:, 1], outAER[:, 0], '.', label='actual')
	plt.title('Training Loss')
	plt.xlabel('time')
	plt.ylabel('neuron ID')
	plt.legend()

	plt.show()
