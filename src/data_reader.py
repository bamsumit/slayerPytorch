import os
import csv
import numpy as np
import yaml
import torch
from collections import namedtuple

from torch.utils.data import Dataset

np_event_type = [('x', np.uint16), ('y', np.uint16), ('p', np.uint8), ('ts', np.uint32)]
DataSample = namedtuple('DataSample', ['number', 'label'])

# Consider dictionary for easier iteration and better scalability
class SlayerParams(object):

	def __init__(self, parameter_file_path):
		with open(parameter_file_path, 'r') as param_file:
			self.parameters = yaml.load(param_file)

	# Allow dictionary like access
	def __getitem__(self, key):
		return self.parameters[key]

	def __setitem__(self, key, value):
		self.parameters[key] = value

class DataReader(Dataset):

	def __init__(self, dataset_folder, training_file, net_params, device=torch.device('cpu')):
		self.EVENT_BIN_SIZE = 5
		self.net_params = net_params
		# Get files in folder
		self.dataset_path = dataset_folder
		self.training_samples = self.read_labels_file(dataset_folder + training_file)
		self.device = device

	# Pytorch Dataset functions
	def __len__(self):
		return len(self.training_samples)

	# TODO refactor n_timesteps and remove repeated uses
	def __getitem__(self, index):
		n_timesteps = int((self.net_params['t_end'] - self.net_params['t_start']) / self.net_params['t_s'])
		data = torch.tensor(self.read_and_bin_input_file(self.training_samples[index]), device=self.device)
		data = data.reshape(1, self.net_params['input_channels'], self.net_params['input_x'], self.net_params['input_y'], n_timesteps)
		return (data, self.training_samples[index].label)
		
	def read_labels_file(self, file):
		# Open CSV file that describes our samples
		labels = []
		with open(file, 'r') as testing_file:
			reader = csv.reader(testing_file, delimiter='\t')
			# Skip header
			next(reader, None)
			for line in reader:
				# TODO cleanup this using map
				labels.append(DataSample(int(line[0]), int(line[1])))
		return labels

	def process_event(self, raw_bytes):
		ts = int.from_bytes(raw_bytes[2:], byteorder='big') & 0x7FFFFF
		return (raw_bytes[0], raw_bytes[1], raw_bytes[2] >> 7, ts)

	# TODO optimize, remove iteration
	# TODO make generic to 1d and 2d spike files
	def read_and_bin_input_file(self, sample):
		file_name = self.dataset_path + str(sample.number) + ".bs2"
		n_inputs = self.net_params['input_x'] * self.net_params['input_y'] * self.net_params['input_channels']
		n_timesteps = int((self.net_params['t_end'] - self.net_params['t_start']) / self.net_params['t_s'])
		# Preallocate numpy array
		binned_array = np.zeros((n_inputs, n_timesteps), dtype=np.float32)
		with open(file_name, 'rb') as input_file:
			for (index, raw_spike) in enumerate(iter(lambda: input_file.read(self.EVENT_BIN_SIZE), b'')):
				(ev_x, ev_y, ev_p, ev_ts) = self.process_event(raw_spike)
				time_position = int(ev_ts / (self.net_params['time_unit'] * self.net_params['t_s']))
				# TODO do truncation if ts over t_end, checks on x and y
				input_position = ev_p * (self.net_params['input_x'] * self.net_params['input_y']) + (ev_y * self.net_params['input_x']) + ev_x
				binned_array[input_position, time_position] = 1.0 / self.net_params['t_s']
		return binned_array

	# Unclear whether this will really be needed, read target spikes in csv format
	def read_output_spikes(self, filename):
		return np.genfromtxt(self.dataset_path + filename, delimiter=",")