import os
from data_reader import DataReader, SlayerParams
from slayer_train import SlayerTrainer, SpikeFunc, SpikeLinear
import torch.nn as nn
import torch
import unittest
from torch.utils.data import DataLoader
from datetime import datetime

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
# Speedup if input size is constant
torch.backends.cudnn.benchmark = True

class NMNISTNet(nn.Module):

    def __init__(self, net_params, weights_init = [0.5,1,1], device=torch.device('cpu')):
        super(NMNISTNet, self).__init__()
        self.net_params = net_params
        self.Ns = int((net_params['t_end'] - net_params['t_start']) / net_params['t_s'])
        self.trainer = SlayerTrainer(net_params, device)
        self.input_srm = self.trainer.calculate_srm_kernel()
        self.srm = self.input_srm[0,0,:,:,:].reshape(1,1,1,1,self.input_srm.shape[-1])
        self.ref = self.trainer.calculate_ref_kernel()
        # Emulate a fully connected 34x34x2 -> 500
        self.fc1 = SpikeLinear(net_params['input_x']*net_params['input_y']*net_params['input_channels'], 500).to(device)
        nn.init.normal_(self.fc1.weight, mean=0, std=weights_init[0])
        # Emulate a fully connected 500 -> 500
        self.fc2 = SpikeLinear(500,500).to(device)
        nn.init.normal_(self.fc2.weight, mean=0, std=weights_init[1])
        # Output layer
        self.fc3 = SpikeLinear(500, net_params['num_classes']).to(device)
        nn.init.normal_(self.fc3.weight, mean=0, std=weights_init[2])
        self.device=device

    def forward(self, x):
        # Apply srm to input spikes
        x = self.trainer.apply_srm_kernel(x, self.input_srm)
        # Flatten the array
        x = x.reshape((self.net_params['batch_size'], 1, 1, self.net_params['input_x']*self.net_params['input_y']*self.net_params['input_channels'], self.Ns))
        # Linear + activation
        x = self.fc1(x)
        x = SpikeFunc.apply(x, self.net_params, self.ref, self.net_params['af_params']['sigma'][0], self.device)
        # Apply srm to middle layer spikes
        x = self.trainer.apply_srm_kernel(x.view(self.net_params['batch_size'],1,1,500,self.Ns), self.srm)
        x = x.reshape((self.net_params['batch_size'], 1, 1, 500, self.Ns))
        # # Apply second layer
        x = SpikeFunc.apply(self.fc2(x), self.net_params, self.ref, self.net_params['af_params']['sigma'][1], self.device)
        # Srm to second hidden layer
        x = self.trainer.apply_srm_kernel(x.view(self.net_params['batch_size'],1,1,500,self.Ns), self.srm)
        x = x.reshape((self.net_params['batch_size'], 1, 1, 500, self.Ns))
        # Output layer
        x = SpikeFunc.apply(self.fc3(x), self.net_params, self.ref, self.net_params['af_params']['sigma'][2], self.device)
        return x

class TestNMNISTTraining(unittest.TestCase):

    def setUp(self):
        self.FILES_DIR = "/fast/sumit/NMNIST_34/"
        self.net_params = SlayerParams(CURRENT_TEST_DIR + "/../test/test_files/NMNISTsmall/" + "parameters.yaml")
        # self.net_params['batch_size'] = 20
        self.cuda = torch.device('cuda')
        self.reader = DataReader(self.FILES_DIR, "train.txt", self.net_params, self.cuda)
        self.test_reader = DataReader(self.FILES_DIR, "test.txt", self.net_params, self.cuda, file_offset=60001)
        self.trainer = SlayerTrainer(self.net_params, self.cuda)
        self.net = NMNISTNet(self.net_params, device=self.cuda)
        self.train_loader = DataLoader(dataset=self.reader, batch_size=self.net_params['batch_size'], shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=self.test_reader, batch_size=self.net_params['batch_size'], shuffle=False, num_workers=1)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

    def test_nmnist_train(self):
        # Needed for CUDA allocation to work in DataLoader
        torch.multiprocessing.set_start_method("spawn")
        for epoch in range(100):
            correct_classifications = 0
            training_loss = 0
            epoch_t0 = datetime.now()
            for i, data in enumerate(self.train_loader, 0):
                t0 = datetime.now()
                self.optimizer.zero_grad()
                minibatch, des_spikes, labels = data
                minibatch = minibatch.reshape(self.net_params['batch_size'],2,1,34*34,350)
                output = self.net(minibatch)
                correct_classifications += self.trainer.get_accurate_classifications(output, labels)
                loss = self.trainer.calculate_l2_loss_classification(output, des_spikes)
                training_loss += loss.data
                loss.backward()
                self.optimizer.step()
                print(i, ":", (datetime.now() - t0).total_seconds())
            print("Epoch n.", epoch)
            print("Epoch time", (datetime.now() - epoch_t0).total_seconds())
            epoch_t0 = datetime.now()
            print("Training accuracy: ", correct_classifications / (len(self.train_loader) * self.net_params['batch_size']))
            print("Training loss: ", training_loss.data / (len(self.train_loader) * self.net_params['batch_size']))
            correct_classifications = 0
            testing_loss = 0
            for i, data in enumerate(self.test_loader, 0):
                minibatch, des_spikes, labels = data
                minibatch = minibatch.reshape(self.net_params['batch_size'],2,1,34*34,350)
                output = self.net(minibatch)
                correct_classifications += self.trainer.get_accurate_classifications(output, labels)
                testing_loss += self.trainer.calculate_l2_loss_classification(output, des_spikes).data
            print("Testing accuracy: ", correct_classifications / len(self.test_loader))
            print("Testing loss: ", testing_loss.data / len(self.test_loader))

    # def test_profile_nmnist(self):
    #     # Needed for CUDA allocation to work in DataLoader
    #     torch.multiprocessing.set_start_method("spawn")
    #     for i, data in enumerate(self.train_loader, 0):
    #         t0 = datetime.now()
    #         self.optimizer.zero_grad()
    #         minibatch, des_spikes, labels = data
    #         minibatch = minibatch.reshape(10,2,1,34*34,350)
    #         for i in range(10):
    #             output = self.net(minibatch)
    #             loss = self.trainer.calculate_l2_loss_classification(output, des_spikes)
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #         with torch.cuda.profiler.profile():
    #             output = self.net(minibatch)
    #             loss = self.trainer.calculate_l2_loss_classification(output, des_spikes)
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #             with torch.autograd.profiler.emit_nvtx():
    #                 for i in range(500):
    #                     output = self.net(minibatch)
    #                     loss = self.trainer.calculate_l2_loss_classification(output, des_spikes)
    #                     loss.backward()
    #                     self.optimizer.step()
    #                 return


if __name__ == '__main__':
    unittest.main()