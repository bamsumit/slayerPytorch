# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import iterable_float_pair_comparator, iterable_int_pair_comparator, is_array_equal_to_file
from slayer_train import SlayerTrainer
import unittest
import os
from itertools import zip_longest
import operator
import numpy as np
import torch

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
		self.FILES_DIR = CURRENT_TEST_DIR + "/test_files/torch_validate/forward_prop/"
		self.net_params = SlayerParams(self.FILES_DIR + "parameters.yaml")
		self.trainer = SlayerTrainer(self.net_params)
		self.srm = self.trainer.calculate_srm_kernel()
		self.ref = self.trainer.calculate_ref_kernel()
		self.compare_params = {'FLOAT_EPS_TOL' : 5e-2}
		self.gtruth = self.read_forwardprop_gtruth()

	def read_forwardprop_gtruth(self):
		gtruth = {}
		for file in os.listdir(self.FILES_DIR):
			if file.endswith('.csv'):
				gtruth[file[0:-4]] = torch.from_numpy(np.genfromtxt(self.FILES_DIR + file, delimiter=",", dtype=np.float32))
		return gtruth

	def test_ref_kernel_generation(self):
		self.assertEqual(len(self.ref), 110)
		# Check values
		ref_gtruth = np.genfromtxt(self.FILES_DIR + "../refractory_kernel.csv", delimiter=",", dtype=np.float32)
		self.assertTrue(iterable_float_pair_comparator(self.ref, ref_gtruth, self.compare_params))

	def test_forward_prop_single_sample(self):
		# Apply SRM to input spikes
		a1 = self.trainer.apply_srm_kernel(self.gtruth['s1'].reshape(1,1,1,250,501), self.srm)
		# Check value
		self.assertTrue(iterable_float_pair_comparator(a1.flatten(), self.gtruth['a1'].flatten(), self.compare_params))
		# Calculate membrane potential and spikes
		(u2, s2) = self.trainer.calculate_membrane_potentials(a1.reshape(250,501), self.gtruth['W12'].reshape(25,250), self.ref, 
			self.net_params['af_params']['sigma'][1])
		# Check values
		self.assertTrue(iterable_float_pair_comparator(u2.flatten(), self.gtruth['u2'].flatten(), self.compare_params))
		self.assertTrue(iterable_int_pair_comparator(s2.flatten(), self.gtruth['s2'].flatten(), self.compare_params))
		# Just for safety do next layer
		a2 = self.trainer.apply_srm_kernel(s2.reshape(1,1,1,25,501), self.srm)
		self.assertTrue(iterable_float_pair_comparator(a2.flatten(), self.gtruth['a2'].flatten(), self.compare_params))
		(u3, s3) = self.trainer.calculate_membrane_potentials(a2.reshape(25,501), self.gtruth['W23'].reshape(1,25), self.ref, 
			self.net_params['af_params']['sigma'][2])
		self.assertTrue(iterable_int_pair_comparator(s3.flatten(), self.gtruth['s3'].flatten(), self.compare_params))


if __name__ == '__main__':
	unittest.main()