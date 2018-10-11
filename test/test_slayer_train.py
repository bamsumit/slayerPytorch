# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import iterable_float_pair_comparator, iterable_int_pair_comparator, is_array_equal_to_file
from slayer_train import SlayerTrainer, SpikeFunc, SlayerNet
import unittest
import os
from itertools import zip_longest
import operator
import numpy as np
import torch

def read_gtruth_folder(folder):
		gtruth = {}
		for file in os.listdir(folder):
			if file.endswith('.csv'):
				gtruth[file[0:-4]] = torch.from_numpy(np.genfromtxt(folder + file, delimiter=",", dtype=np.float32))
		return gtruth

class TestSlayerSRM(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.trainer = SlayerTrainer(self.net_params)
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", "test100.txt", self.net_params)
		self.FLOAT_EPS_TOL = 1e-3 # Tolerance for floating point equality
		self.srm = self.trainer.calculate_srm_kernel()

	def test_srm_kernel_truncated_int_tend(self):
		self.trainer.net_params['t_end'] = 3
		truncated_srm = self.trainer.calculate_srm_kernel()
		self.assertEqual(truncated_srm.shape, (self.net_params['input_channels'], self.net_params['input_channels'], 1, 1, 2 * self.trainer.net_params['t_end'] - 1))

	def test_srm_kernel_not_truncated(self):
		# Calculated manually
		max_abs_diff = 0
		# The first are prepended 0s for causality
		srm_g_truth = [ 0, 0.0173512652, 0.040427682, 0.0915781944, 0.1991482735, 0.4060058497, 0.7357588823, 1, 0,
						0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(self.srm.shape, (self.net_params['input_channels'], self.net_params['input_channels'], 1, 1, len(srm_g_truth)))
		# We want 0 in every non i=j line, and equal to g_truth in every i=j line
		for out_ch in range(self.net_params['input_channels']):
			for in_ch in range(self.net_params['input_channels']):
				cur_max = 0
				if out_ch == in_ch:
					cur_max = max([abs(v[0] - v[1]) for v in zip_longest(self.srm[out_ch, in_ch, :, :, :].flatten(), srm_g_truth)])
				else:
					cur_max = max(abs(self.srm[out_ch, in_ch, :, :, :].flatten()))
				max_abs_diff = cur_max if cur_max > max_abs_diff else max_abs_diff
		max_abs_diff = max([abs(v[0] - v[1]) for v in zip(self.srm.flatten(), srm_g_truth)])
		self.assertTrue(max_abs_diff < self.FLOAT_EPS_TOL)

	def test_convolution_with_srm_minimal(self):
		input_spikes = torch.FloatTensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0, 
										   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		input_spikes = input_spikes.reshape((1,2,1,1,21))
		srm_response = self.trainer.apply_srm_kernel(input_spikes, self.srm)
		self.assertEqual(srm_response.shape, (1,2,1,1,21))
		# Full test is the one below

	def test_convolution_with_srm_kernel(self):
		input_spikes = self.reader.get_minibatch(1)
		srm_response = self.trainer.apply_srm_kernel(input_spikes, self.srm)
		self.assertTrue(is_array_equal_to_file(srm_response.reshape((2312,350)), CURRENT_TEST_DIR + "/test_files/torch_validate/1_spike_response_signal.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : self.FLOAT_EPS_TOL}))

	def test_srm_with_ts_normalization(self):
		# Not very user friendly API, consider refactoring
		self.reader.net_params['t_s'] = 2
		self.trainer.net_params['t_s'] = 2
		srm_downsampled = self.trainer.calculate_srm_kernel()
		input_spikes = self.reader.get_minibatch(1)
		srm_response = self.trainer.apply_srm_kernel(input_spikes, srm_downsampled)
		# np.savetxt("debug_ts2.txt", srm_response.reshape((2312,175)).numpy())
		self.assertTrue(is_array_equal_to_file(srm_response.reshape((2312,175)), CURRENT_TEST_DIR + "/test_files/torch_validate/1_spike_response_signal_ts2.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : self.FLOAT_EPS_TOL}))

class TestForwardProp(unittest.TestCase):

	def setUp(self):
		self.FILES_DIR = CURRENT_TEST_DIR + "/test_files/torch_validate/"
		self.net_params = SlayerParams(self.FILES_DIR + "parameters.yaml")
		self.trainer = SlayerTrainer(self.net_params)
		self.srm = self.trainer.calculate_srm_kernel()
		self.ref = self.trainer.calculate_ref_kernel()
		self.spike_func = SpikeFunc()
		self.compare_params = {'FLOAT_EPS_TOL' : 5e-2}
		self.fprop_gtruth = read_gtruth_folder(self.FILES_DIR + "forward_prop/")
		self.bprop_gtruth = read_gtruth_folder(self.FILES_DIR + "back_prop/")

	def test_ref_kernel_generation(self):
		self.assertEqual(len(self.ref), 110)
		# Check values
		ref_gtruth = np.genfromtxt(self.FILES_DIR + "refractory_kernel.csv", delimiter=",", dtype=np.float32)
		self.assertTrue(iterable_float_pair_comparator(self.ref, ref_gtruth, self.compare_params))

	def test_membrane_potential_shape(self):
		pots = self.spike_func.apply_weights(self.fprop_gtruth['a1'].reshape(1,250,1,1,501), self.fprop_gtruth['W12'].reshape(25,250,1,1,1))
		s2 = torch.zeros(pots.shape, dtype=torch.float32)
		(u2, s2) = self.spike_func.calculate_membrane_potentials(pots, s2, self.net_params, self.ref, self.net_params['af_params']['sigma'][0])
		self.assertEqual(u2.shape, (1,25,1,1,501))


	def test_forward_prop_single_sample(self):
		# Apply SRM to input spikes
		a1 = self.trainer.apply_srm_kernel(self.fprop_gtruth['s1'].reshape(1,1,1,250,501), self.srm)
		# Check value
		self.assertTrue(iterable_float_pair_comparator(a1.flatten(), self.fprop_gtruth['a1'].flatten(), self.compare_params))
		# Calculate membrane potential and spikes
		pots = self.spike_func.apply_weights(a1.reshape(1,250,1,1,501), self.fprop_gtruth['W12'].reshape(25,250,1,1,1))
		s2 = torch.zeros(pots.shape, dtype=torch.float32)
		(u2, s2) = self.spike_func.calculate_membrane_potentials(pots, s2, self.net_params, self.ref, self.net_params['af_params']['sigma'][0])
		# Check values
		self.assertTrue(iterable_float_pair_comparator(u2.flatten(), self.fprop_gtruth['u2'].flatten(), self.compare_params))
		self.assertTrue(iterable_int_pair_comparator(s2.flatten(), self.fprop_gtruth['s2'].flatten(), self.compare_params))
		# Just for safety do next layer
		a2 = self.trainer.apply_srm_kernel(s2.reshape(1,1,1,25,501), self.srm)
		self.assertTrue(iterable_float_pair_comparator(a2.flatten(), self.fprop_gtruth['a2'].flatten(), self.compare_params))
		pots = self.spike_func.apply_weights(a2.reshape(1,25,1,1,501), self.fprop_gtruth['W23'].reshape(1,25,1,1,1))
		s3 = torch.zeros(pots.shape, dtype=torch.float32)
		(u3, s3) = self.spike_func.calculate_membrane_potentials(pots, s3, self.net_params, self.ref, self.net_params['af_params']['sigma'][1])
		self.assertTrue(iterable_int_pair_comparator(s3.flatten(), self.fprop_gtruth['s3'].flatten(), self.compare_params))
		self.assertTrue(iterable_int_pair_comparator(u3.flatten(), self.fprop_gtruth['u3'].flatten(), self.compare_params))
		# And final activations
		a3 = self.trainer.apply_srm_kernel(s3.reshape(1,1,1,1,501), self.srm)
		self.assertTrue(iterable_float_pair_comparator(a3.flatten(), self.fprop_gtruth['a3'].flatten(), self.compare_params))

	def test_spiketrain_error_calculation(self):
		# Error is difference of the activations in the last layer, calculate desired activations first
		des_a = self.trainer.apply_srm_kernel(self.bprop_gtruth['des_s'].reshape(1,1,1,1,501), self.srm)
		self.assertTrue(iterable_float_pair_comparator(des_a.flatten(), self.bprop_gtruth['des_a'].flatten(), self.compare_params))
		error = self.trainer.calculate_error_spiketrain(self.fprop_gtruth['a3'], des_a)
		self.assertTrue(iterable_float_pair_comparator(error.flatten(), self.bprop_gtruth['e3'].flatten(), self.compare_params))

	def test_pdf_func(self):
		pdf = torch.zeros(self.fprop_gtruth['u3'].shape)
		pdf = self.spike_func.calculate_pdf(self.fprop_gtruth['u3'], pdf, self.net_params['af_params']['theta'],
			self.net_params['pdf_params']['tau'], self.net_params['pdf_params']['scale'])
		self.assertTrue(iterable_float_pair_comparator(pdf.flatten(), self.bprop_gtruth['rho3'].flatten(), self.compare_params))

	def test_l2_loss(self):
		l2loss = self.trainer.calculate_l2_loss(self.fprop_gtruth['a3'], self.bprop_gtruth['des_a'])
		self.assertTrue(abs(l2loss - 7.81225) < self.compare_params['FLOAT_EPS_TOL'])

# from torchviz import make_dot

class TestSlayerNet(unittest.TestCase):

	def setUp(self):
		self.FILES_DIR = CURRENT_TEST_DIR + "/test_files/torch_validate/"
		self.net_params = SlayerParams(self.FILES_DIR + "parameters.yaml")
		self.net = SlayerNet(self.net_params)
		self.fprop_gtruth = read_gtruth_folder(self.FILES_DIR + "forward_prop/")
		self.bprop_gtruth = read_gtruth_folder(self.FILES_DIR + "back_prop/")
		self.trainer = SlayerTrainer(self.net_params)
		self.srm = self.trainer.calculate_srm_kernel()
		self.input_spikes = self.fprop_gtruth['s1'].reshape(1,1,1,250,501)
		des_spikes = self.bprop_gtruth['des_s'].reshape(1,1,1,1,501)
		self.des_activations = self.trainer.apply_srm_kernel(des_spikes, self.srm)
		self.compare_params = {'FLOAT_EPS_TOL' : 2e-3}

	def test_forward_pass(self):
		self.net.fc1.weight = torch.nn.Parameter(self.fprop_gtruth['W12'].reshape(25,1,1,250,1))
		self.net.fc2.weight = torch.nn.Parameter(self.fprop_gtruth['W23'].reshape(1,1,1,25,1))
		out_spikes = self.net.forward(self.input_spikes)
		self.assertEqual(out_spikes.shape, (1,1,1,1,501))
		self.assertTrue(iterable_float_pair_comparator(self.fprop_gtruth['s3'].flatten(), out_spikes.flatten(), self.compare_params))

	def test_gradients_calculation(self):
		# Assign weights manually
		self.net.fc1.weight = torch.nn.Parameter(self.fprop_gtruth['W12'].reshape(25,1,1,250,1))
		self.net.fc2.weight = torch.nn.Parameter(self.fprop_gtruth['W23'].reshape(1,1,1,25,1))
		# Forward prop
		out_spikes = self.net.forward(self.input_spikes)
		# Activations
		output_activations = self.trainer.apply_srm_kernel(out_spikes, self.srm)
		# Calculate l2 loss
		l2loss = self.trainer.calculate_l2_loss(output_activations, self.des_activations)
		l2loss.backward()
		self.assertTrue(iterable_float_pair_comparator(self.net.fc2.weight.grad.flatten(), self.bprop_gtruth['Wgrad2'].flatten(), self.compare_params))
		self.assertTrue(iterable_float_pair_comparator(self.net.fc1.weight.grad.flatten(), self.bprop_gtruth['Wgrad1'].flatten(), self.compare_params))

	# def test_training(self):
	# 	self.net.fc1.weight = torch.nn.Parameter(self.fprop_gtruth['W12'].reshape(25,1,1,250,1))
	# 	self.net.fc2.weight = torch.nn.Parameter(self.fprop_gtruth['W23'].reshape(1,1,1,25,1))
	# 	MAX_ITER = 20000
	# 	optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
	# 	for i in range(MAX_ITER):
	# 		optimizer.zero_grad()
	# 		outputs = self.net(self.input_spikes)
	# 		output_activations = self.trainer.apply_srm_kernel(outputs, self.srm)
	# 		loss = self.trainer.calculate_l2_loss(output_activations, self.des_activations)
	# 		print("iteration n.", i, "loss = ", float(loss.detach()))
	# 		loss.backward()
	# 		if loss == 0:
	# 			break
	# 		optimizer.step()


class TestCustomCudaKernel(unittest.TestCase):

	def setUp(self):
		self.FILES_DIR = CURRENT_TEST_DIR + "/test_files/torch_validate/"
		self.net_params = SlayerParams(self.FILES_DIR + "parameters.yaml")
		self.fprop_gtruth = read_gtruth_folder(self.FILES_DIR + "forward_prop/")
		self.bprop_gtruth = read_gtruth_folder(self.FILES_DIR + "back_prop/")
		self.cuda = torch.device('cuda')
		self.trainer = SlayerTrainer(self.net_params, self.cuda)
		self.compare_params = {'FLOAT_EPS_TOL' : 5e-2}
		self.ref = self.trainer.calculate_ref_kernel()

	def test_cuda_spikefunction(self):
		input_pots = SpikeFunc.apply_weights(self.fprop_gtruth['a1'].reshape(1,1,1,250,501), self.fprop_gtruth['W12'].reshape(25,1,1,250,1)).to(self.cuda)
		pots_gtruth = self.fprop_gtruth['u2_sigma0'].reshape(1,25,1,1,501)
		spikes_gtruth = self.fprop_gtruth['s2_sigma0'].reshape(1,1,1,25,501)
		spikes_out = torch.zeros(spikes_gtruth.shape, dtype=torch.float32).to(self.cuda)
		(u2, s2) = SpikeFunc.get_spikes_cuda(input_pots, spikes_out, self.ref, self.net_params)
		# Check for correctness
		self.assertTrue(iterable_float_pair_comparator(u2.flatten(), pots_gtruth.flatten(), self.compare_params))
		self.assertTrue(iterable_float_pair_comparator(s2.flatten(), spikes_gtruth.flatten(), self.compare_params))

	# def test_training(self):
	# 	net = SlayerNet(self.net_params, device=self.cuda)
	# 	input_spikes = self.fprop_gtruth['s1'].reshape(1,1,1,250,501).to(self.cuda)
	# 	des_spikes = self.bprop_gtruth['des_s'].reshape(1,1,1,1,501).to(self.cuda)
	# 	srm = self.trainer.calculate_srm_kernel()
	# 	des_activations = self.trainer.apply_srm_kernel(des_spikes, srm)
	# 	MAX_ITER = 20000
	# 	optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
	# 	for i in range(MAX_ITER):
	# 		optimizer.zero_grad()
	# 		outputs = net(input_spikes)
	# 		output_activations = self.trainer.apply_srm_kernel(outputs, srm)
	# 		loss = self.trainer.calculate_l2_loss(output_activations, des_activations)
	# 		print("iteration n.", i, "loss = ", float(loss.detach()))
	# 		loss.backward()
	# 		if loss == 0:
	# 			break
	# 		optimizer.step()



if __name__ == '__main__':
	unittest.main()