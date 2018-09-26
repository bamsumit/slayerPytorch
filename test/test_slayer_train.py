# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import iterable_float_pair_comparator, is_array_equal_to_file
from slayer_train import SlayerTrainer
import unittest
import numpy as np

class TestSlayerTrainKernels(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.trainer = SlayerTrainer(self.net_params)
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", "test100.txt", self.net_params)
		self.minibatch_size = 12
		self.FLOAT_EPS_TOL = 1e-9 # Tolerance for floating point equality

	def test_srm_kernel_truncated_int_tend(self):
		self.trainer.net_params['t_end'] = 3
		srm = self.trainer.calculate_srm_kernel()
		self.assertEqual(len(srm), 3)

	def test_srm_kernel_not_truncated(self):
		srm = self.trainer.calculate_srm_kernel()
		# Calculated manually
		srm_g_truth = [0, 1, 0.7357588823, 0.4060058497, 0.1991482735, 0.0915781944, 0.040427682, 0.0173512652]
		self.assertEqual(len(srm), len(srm_g_truth))
		max_abs_diff = max([abs(v[0] - v[1]) for v in zip(srm, srm_g_truth)])
		self.assertTrue(max_abs_diff < self.FLOAT_EPS_TOL)

	def test_convolution_with_srm_kernel(self):
		srm = self.trainer.calculate_srm_kernel()
		input_spikes = self.reader.read_and_bin(self.reader.training_samples[0])
		srm_response = self.trainer.apply_srm_kernel(input_spikes, srm)
		self.assertTrue(is_array_equal_to_file(srm_response, CURRENT_TEST_DIR + "/test_files/torch_validate/1_spike_response_signal.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : self.FLOAT_EPS_TOL}))


if __name__ == '__main__':
	unittest.main()