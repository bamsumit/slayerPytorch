# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import iterable_float_pair_comparator, is_array_equal_to_file
from slayer_train import SlayerTrainer
import unittest
from itertools import zip_longest
import numpy as np
import torch

class TestSlayerTrainKernels(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.trainer = SlayerTrainer(self.net_params)
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", "test100.txt", self.net_params)
		self.FLOAT_EPS_TOL = 1e-3 # Tolerance for floating point equality

	def test_srm_kernel_truncated_int_tend(self):
		self.trainer.net_params['t_end'] = 3
		srm = self.trainer.calculate_srm_kernel()
		self.assertEqual(srm.shape, (self.net_params['input_channels'], self.net_params['input_channels'], 1, 1, 2 * self.trainer.net_params['t_end'] - 1))

	def test_srm_kernel_not_truncated(self):
		srm = self.trainer.calculate_srm_kernel()
		# Calculated manually
		max_abs_diff = 0
		# The first are prepended 0s for causality
		srm_g_truth = [ 0, 0.0173512652, 0.040427682, 0.0915781944, 0.1991482735, 0.4060058497, 0.7357588823, 1, 0,
						0, 0, 0, 0, 0, 0, 0, 0]
		self.assertEqual(srm.shape, (self.net_params['input_channels'], self.net_params['input_channels'], 1, 1, len(srm_g_truth)))
		# We want 0 in every non i=j line, and equal to g_truth in every i=j line
		for out_ch in range(self.net_params['input_channels']):
			for in_ch in range(self.net_params['input_channels']):
				cur_max = 0
				if out_ch == in_ch:
					cur_max = max([abs(v[0] - v[1]) for v in zip_longest(srm[out_ch, in_ch, :, :, :].flatten(), srm_g_truth)])
				else:
					cur_max = max(abs(srm[out_ch, in_ch, :, :, :].flatten()))
				max_abs_diff = cur_max if cur_max > max_abs_diff else max_abs_diff
		max_abs_diff = max([abs(v[0] - v[1]) for v in zip(srm.flatten(), srm_g_truth)])
		self.assertTrue(max_abs_diff < self.FLOAT_EPS_TOL)

	def test_convolution_with_srm_minimal(self):
		srm = self.trainer.calculate_srm_kernel()
		input_spikes = torch.FloatTensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0, 
										   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		input_spikes = input_spikes.reshape((1,2,1,1,21))
		srm_response = self.trainer.apply_srm_kernel(input_spikes, srm)
		self.assertEqual(srm_response.shape, (1,2,1,1,21))
		# Full test is the one below

	def test_convolution_with_srm_kernel(self):
		srm = self.trainer.calculate_srm_kernel()
		input_spikes = self.reader.get_minibatch(1)
		srm_response = self.trainer.apply_srm_kernel(input_spikes, srm)
		self.assertTrue(is_array_equal_to_file(srm_response.reshape((2312,350)), CURRENT_TEST_DIR + "/test_files/torch_validate/1_spike_response_signal.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : self.FLOAT_EPS_TOL}))


if __name__ == '__main__':
	unittest.main()