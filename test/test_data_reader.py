# Add to path
import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

sys.path.append(CURRENT_TEST_DIR + "/../src")

from data_reader import DataReader
import csv
import unittest

def matlab_equal_to_python_event(matlab_event, python_event):
	# Cast to avoid type problems
	matlab_event = [int(e) for e in matlab_event]
	python_event = [int(e) for e in python_event]
	# Matlab is 1 indexed, Python is 0 indexed
	return ((matlab_event[0] == (python_event[0] + 1)) and (matlab_event[1] == (python_event[1] + 1)) and
		(matlab_event[2] == (python_event[2] + 1)) and (matlab_event[3] == (python_event[3])))

class TestDataReaderFolders(unittest.TestCase):

	def test_open_nonexisting_folder(self):
		self.assertRaises(FileNotFoundError, DataReader, "nonexisting_folder")

	def test_open_valid_folder(self):
		try:
			reader = DataReader(CURRENT_TEST_DIR + "/test_files/input_data")
		except FileNotFoundError:
			self.fail("Valid input folder not found")

	def test_number_of_files_invalid_folder(self):
		# Should return 0 files valid
		reader = DataReader(CURRENT_TEST_DIR + "/test_files/input_validate")
		self.assertEqual(len(reader.input_files), 0)

	def test_input_files_ordering(self):
		file_folder = CURRENT_TEST_DIR + "/test_files/input_data/"
		reader = DataReader(file_folder)
		self.assertEqual(reader.input_files[0], file_folder + '1.bs2')


class TestDataReaderFunc(unittest.TestCase):

	def setUp(self):
		self.reader = DataReader(CURRENT_TEST_DIR + "/test_files/input_data")

	def test_number_of_files_valid_folder(self):
		self.assertEqual(len(self.reader.input_files), 2)

	def test_read_invalid_input_file(self):
		self.assertRaises(FileNotFoundError, self.reader.read_file, "nonexisting_file.garbage")

	def test_process_event(self):
		# Actually first line of test file
		raw_bytes = bytes.fromhex('121080037d')
		# Everything is zero indexed in python, except time
		event = (19,17,2,893)
		self.assertTrue(matlab_equal_to_python_event(event, self.reader.process_event(raw_bytes)))

	# Check proper I/O
	def test_read_input_file(self):
		ev_array = self.reader.read_file(self.reader.input_files[0])
		with open(CURRENT_TEST_DIR + "/test_files/input_validate/1.csv", 'r') as csvfile:
			reader = csv.reader(csvfile)
			# Skip header
			next(reader, None)
			for (g_truth, read_r) in zip(reader, ev_array):
				self.assertTrue(matlab_equal_to_python_event(g_truth, read_r))


if __name__ == '__main__':
	unittest.main()