# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader, SlayerParams
from testing_utilities import is_array_equal_to_file, iterable_float_pair_comparator
import csv
import unittest
import numpy as np

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

	def test_input_files_ordering(self):
		file_folder = CURRENT_TEST_DIR + "/test_files/NMNISTsmall/"
		reader = DataReader(file_folder, "train1K.txt", self.net_params)
		self.assertEqual(reader.dataset_path + str(reader.training_samples[0].number) + ".bs2", file_folder + '1.bs2')

	# def test_init_invalid_network_params(self):
	# 	invalid_params = SlayerParams()
	# 	self.assertRaises(ValueError, DataReader, CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", invalid_params)


class TestDataReaderInputFile(unittest.TestCase):

	def setUp(self):
		self.net_params = SlayerParams(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/" + "parameters.yaml")
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/NMNISTsmall/", "train1K.txt", self.net_params)

	def test_number_of_files_valid_folder(self):
		self.assertEqual(len(self.reader.training_samples), NMNIST_SIZE)

	def test_process_event(self):
		# Actually first line of test file
		raw_bytes = bytes.fromhex('121080037d')
		# Everything is zero indexed in python, except time
		event = (19,17,2,893)
		self.assertTrue(matlab_equal_to_python_event(event, self.reader.process_event(raw_bytes)))

	# Check proper I/O
	def test_read_input_file(self):
		ev_array = self.reader.read_input_file(self.reader.training_samples[0])
		self.assertTrue(is_array_equal_to_file(ev_array, CURRENT_TEST_DIR + "/test_files/input_validate/1_raw_spikes.csv", has_header=True, compare_function=matlab_equal_to_python_event))

	def test_spikes_binning(self):
		ev_array = self.reader.read_input_file(self.reader.training_samples[0])
		binned_spikes = self.reader.bin_spikes(ev_array)
		self.assertTrue(is_array_equal_to_file(binned_spikes, CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes.csv"))

	def test_high_level_binning(self):
		binned_spikes = self.reader.read_and_bin_np(self.reader.training_samples[0])
		self.assertTrue(is_array_equal_to_file(binned_spikes, CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes.csv"))

	def test_read_input_spikes_ts_normalization(self):
		FLOAT_EPS_TOL = 1e-6
		self.reader.net_params['t_s'] = 2
		binned_spikes = self.reader.read_and_bin_np(self.reader.training_samples[0])
		self.assertTrue(is_array_equal_to_file(binned_spikes, CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes_ts2.csv", 
			compare_function=iterable_float_pair_comparator, comp_params={"FLOAT_EPS_TOL" : FLOAT_EPS_TOL}))

	def test_loaded_label_value(self):
		self.assertEqual(self.reader.training_samples[0].label, 5)

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
		(binned_spikes, label) = self.reader[0]
		self.assertTrue(is_array_equal_to_file(binned_spikes.reshape(2312,350), CURRENT_TEST_DIR + "/test_files/input_validate/1_binned_spikes.csv"))
		self.assertEqual(label, 5)


if __name__ == '__main__':
	unittest.main()