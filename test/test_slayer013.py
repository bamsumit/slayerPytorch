import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import slayerSNN as snn


############################################################################################
# To test the correctness of transposed convolution operation and unpooling operation ######

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

device = torch.device('cuda') 

netParams = snn.params('test_files/nmnistNet.yaml')

slayer = snn.layer(netParams['neuron'], netParams['simulation']).to(device)

# transposed conv operation test

class TestConvTranspose(unittest.TestCase):
    def test(self):
        # (N, C, H, W, D) = (1, 1, 8, 8, 1)
        (N, C, H, W, D) = (4, 8, 100, 100, 500)
        K = 5
        Cin = 3

        convT = slayer.convTranspose(Cin, C, K).to(device)
        inTensor = torch.randn((N, Cin, H, W, D)).to(device)
        weight = convT.weight.cpu().data.numpy().reshape((Cin, C, K, K))
        outGT = torch.zeros((N, C, H+K-1, W+K-1, D)).to(device)

        # print('wgt  :', weight.shape)
        # print('in   :', inTensor.shape)
        # print('outGT:', outGT.shape)

        for c in range(Cin):
            for i in range(K):
                for j in range(K):
                    for k in range(C):
                        outGT[:, k, j:H+j, i:W+i, :] += weight[c, k, j, i] * inTensor[:, c, ...]

        out = convT(inTensor)

        # print('out  :', out.shape)
        
        error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)
        # print('Transposed Conv Error :', error)

        self.assertTrue(error < 1e-6, 'Transposed Conv result (out) must match outGT.')


# prGTint(inTensor.cpu().data)
# print(conv.weight.cpu().data)
# print(out.cpu().data)
# print(outGt.cpu().data)

# unpooling operation

class TestUnpooling(unittest.TestCase):
    def test(self):
        unpool = slayer.unpool(2).to(device)
        inTensor = torch.randn((4, 8, 50, 50, 500)).to(device)
        outGT = torch.zeros((4, 8, 100, 100, 500)).to(device)
        
        outGT[:, :, 0::2, 0::2, :] = inTensor * 1.1 * netParams['neuron']['theta']
        outGT[:, :, 0::2, 1::2, :] = inTensor * 1.1 * netParams['neuron']['theta']
        outGT[:, :, 1::2, 0::2, :] = inTensor * 1.1 * netParams['neuron']['theta']
        outGT[:, :, 1::2, 1::2, :] = inTensor * 1.1 * netParams['neuron']['theta']
        
        out   = unpool(inTensor)

        error = torch.norm(out - outGT).cpu().data.numpy() / torch.numel(out)

        # print('Pool Error :', error)
        self.assertTrue(error < 1e-6, 'Pool result (out) must match outGT.')
    
if __name__ == '__main__':
    unittest.main()
