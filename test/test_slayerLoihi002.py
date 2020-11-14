import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
from slayerSNN import loihi as spikeLayer
from slayerSNN import params as SlayerParams
import torch

refDelay = np.random.randint(16) + 2
numSteps = 200

netDesc = {
    'simulation' : {'Ts': 1, 'tSample': numSteps},
    'neuron' : {
        'type'     : 'LOIHI',
        'vThMant'  : 80,
        'vDecay'   : 128,
        'iDecay'   : 1024,
        'refDelay' : refDelay,
        'wgtExp'   : 0,
        'tauRho'   : 1,
        'scaleRho' : 768,
    },
}

class Neuron():
    def __init__(self, vThMant, vDecay, iDecay, refDelay, wgtExp=0):
        self.vThMant = vThMant
        self.vDecay = vDecay
        self.iDecay = iDecay
        self.refDelay = refDelay
        self.wgtExp = wgtExp
        self.theta = self.vThMant * (1<<6)
        self.weightScale = 1 << (6 + self.wgtExp)
        
    def run(self, wSpikes):
        Ns = wSpikes.shape[-1]
        
        voltage = np.zeros(wSpikes.shape).astype(int)
        current = np.zeros(wSpikes.shape).astype(int)
        spike   = np.zeros(wSpikes.shape).astype(int)
        
        refState = np.zeros((wSpikes.shape[0],)).astype(int)
        
        for t in range(Ns):
            # if t==0:
            #     current[:, t] = self.weightScale * wSpikes[:, t].astype(int)
            #     voltage[:, t] = self.weightScale * wSpikes[:, t].astype(int)
            # else:
            #     currentSign = 2 * (current[:, t-1] > 0) - 1
            #     voltageSign = 2 * (voltage[:, t-1] > 0) - 1
                
            #     current[:, t] = currentSign * ( (currentSign * current[:, t-1] * ( (1<<12) - self.iDecay ) ) >> 12 ) + self.weightScale * wSpikes[:, t].astype(int)
            #     voltage[:, t] = voltageSign * ( (voltageSign * voltage[:, t-1] * ( (1<<12) - self.vDecay ) ) >> 12 ) + current[:, t]
            
            if t==0:
                continue
            currentSign = 2 * (current[:, t-1] > 0) - 1
            voltageSign = 2 * (voltage[:, t-1] > 0) - 1
            
            current[:, t] = currentSign * ( (currentSign * current[:, t-1] * ( (1<<12) - self.iDecay ) ) >> 12 ) + self.weightScale * wSpikes[:, t].astype(int)
            voltage[:, t] = voltageSign * ( (voltageSign * voltage[:, t-1] * ( (1<<12) - self.vDecay ) ) >> 12 ) + current[:, t]
            
            if t>=self.refDelay: 
                refState -= spike[:, t-self.refDelay]
            
            spike[:, t] = (voltage[:, t] > self.theta) * (refState == 0)
            
            # print(refState, current[:, t], voltage[:, t], spike[:, t])
            
            voltage[:, t] *= (1 - spike[:, t]) * (refState == 0)
            refState += spike[:, t]
            
        # post computation value adjustments to match the values
        # this does not work for refDelay more than 1 yet
        voltage = (1-spike)*voltage + spike*self.vDecay
        
        return spike, current, voltage
    
    def spikes(self, wSpikes):
        return self.run(wSpikes)[0]

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

if verbose:
    print('refDelay:', refDelay)

device = torch.device('cuda')

numSrc = 128
numDst = 64

net_params = SlayerParams(dict=netDesc)
weight = np.round((np.random.random((numDst, numSrc))-0.5)*10)*2
inputSpikes = np.random.random((numSrc, numSteps)) > 0.8
# inputSpikes[:,0] = 0

wSpikes = weight.dot(inputSpikes)

slayer = spikeLayer(net_params['neuron'], net_params['simulation']).to(device)

if verbose is True: print('Neuron Threshold =', slayer.neuron['theta'])

neuron = Neuron(net_params['neuron']['vThMant'], net_params['neuron']['vDecay'], net_params['neuron']['iDecay'], net_params['neuron']['refDelay'])
spikeGT, currentGT, voltageGT = neuron.run(wSpikes)
spike, voltage, current = slayer.spikeLoihiFull(torch.FloatTensor(wSpikes).to(device))

class testRefDelay(unittest.TestCase):
    def testSpikes(self):
        error = np.sum((spike.cpu().data.numpy() - spikeGT)**2)
        if verbose is True:     print('Spike value error:', error)
        self.assertTrue(error<1e-3, 'Spikes: cuda ouptut and ground truth must match.')

    def testVoltage(self):
        error = np.sum((voltage.cpu().data.numpy() - voltageGT)**2)
        if verbose is True:     print('Neuron voltage value error:', error)
        self.assertTrue(error<1e-3, 'Voltage: cuda ouptut and ground truth must match.')

    def testCurrent(self):
        error = np.sum((current.cpu().data.numpy() - currentGT)**2)
        if verbose is True:     print('Neuron current value error:', error)
        self.assertTrue(error<1e-3, 'Current: cuda ouptut and ground truth must match.')