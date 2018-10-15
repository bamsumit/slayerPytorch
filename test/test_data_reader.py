# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import is_array_equal_to_file, iterable_float_pair_comparator, iterable_int_pair_comparator
import csv
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

NMNIST_SIZE = 1000
NMNIST_NUM_CLASSES = 10

def matlab_equal_to_python_event(matlab_event, python_event, params = {}):
	# Cast to avoid type problems
	matlab_event = [int(e) for e in matlab_event]
	python_event = [int(e) for e in python_event]
	# Matlab is 1 indexed, Python is 0 indexed
	return ((matlab_event[0] == (python_event[0] + 1)) and (matlab_event[1] == (python_event[1] + 1)) and
		(matlab_event[2] == (python_event[2] + 1)) and (matlab_event[3] == (python_event[3])))

class TestSlayerParamsLoader(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")

	# Just test one
	def test_load(self):
		self.assertEqual(self.net_params['t_end'], 350)

class TestDataReaderFolders(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")

	def test_open_valid_folder(self):
		try:
			reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", self.net_params)
		except FileNotFoundError:
			self.fail("Valid input folder not found")

class TestDataReaderInputFile(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", self.net_params)

	def test_number_of_files_valid_folder(self):
		self.assertEqual(self.reader.num_samples, NMNIST_SIZE)

	def test_process_event(self):
		# Actually first line of test file
		raw_bytes = bytes.fromhex('121080037d')
		# Everything is zero indexed in python, except time
		event = (19,17,2,893)
		self.assertTrue(matlab_equal_to_python_event(event, self.reader.process_event(raw_bytes)))

	def test_spikes_binning(self):
		binned_spikes = self.reader.read_and_bin_input_file(1)
		self.assertTrue(is_array_equal_to_file(binned_spikes, CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes.csv"))

	def test_read_input_spikes_ts_normalization(self):
		FLOAT_EPS_TOL = 1e-6
		self.reader.net_params['t_s'] = 2
		binned_spikes = self.reader.read_and_bin_input_file(1)
		self.assertTrue(is_array_equal_to_file(binned_spikes, CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes_ts2.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : FLOAT_EPS_TOL}))

	def test_loaded_label_value(self):
		# Correct class (5) should have 50 spikes, incorrect should have 10
		self.assertEqual(self.reader.training_samples[0,5], 50)
		self.assertEqual(self.reader.training_samples[0,0], 10)

class TestDataReaderOutputSpikes(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/training/", "train12.txt", self.net_params)

	def test_load_output_spikes(self):
		minibatch_size = 12
		num_time_samples = int((self.net_params['t_end'] - self.net_params['t_start']) / self.net_params['t_s'])
		output_spikes = self.reader.read_output_spikes("test12_output_spikes.csv")
		self.assertEqual(output_spikes.shape, (NMNIST_NUM_CLASSES, minibatch_size * num_time_samples))

class TestPytorchDataset(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", self.net_params)

	def test_len(self):
		self.assertEqual(len(self.reader), 1000)

	def test_getitem(self):
		(binned_spikes, des_spikes, label) = self.reader[0]
		self.assertTrue(is_array_equal_to_file(binned_spikes.reshape(2312,350), CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes.csv"))
		# We need tensors for output labels as well
		self.assertEqual(des_spikes.shape, (10,1,1,1))
		self.assertTrue(iterable_int_pair_comparator(des_spikes, [10, 10, 10, 10, 10, 50, 10, 10, 10, 10]))
		self.assertEqual(label, 5)

	def test_dataloader(self):
		loader = DataLoader(dataset=self.reader, batch_size=self.net_params['batch_size'], shuffle=True, num_workers=2)
		(inputs, des_spikes, label) = next(iter(loader))
		self.assertEqual(inputs.shape, (10,2,34,34,350))
		self.assertEqual(des_spikes.shape, (10,10,1,1,1))
		self.assertEqual(len(label), 10)

class TestReaderCuda(unittest.TestCase):

	def test_tensors_in_cuda(self):
		net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", net_params, device=torch.device('cuda'))
		spikes, des_spikes, labels = reader[0]
		self.assertEqual(spikes.device.type, 'cuda')
		self.assertEqual(des_spikes.device.type, 'cuda')


if __name__ == '__main__':
	unittest.main()