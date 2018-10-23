import os
import csv
import numpy as np
import yaml
import torch

from torch.utils.data import Dataset

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

	def __init__(self, dataset_folder, training_file, net_params, device=torch.device('cuda'), file_offset=1):
		self.EVENT_BIN_SIZE = 5
		self.net_params = net_params
		# Get files in folder
		self.dataset_path = dataset_folder
		training_samples, self.num_samples, labels = self.read_labels_file(dataset_folder + training_file)
		self.training_samples = torch.tensor(training_samples, device=device, dtype=torch.float32)
		self.labels = torch.tensor(labels, device=device, dtype=torch.float32)
		self.training_samples = self.training_samples.reshape(self.num_samples, net_params['num_classes'], 1, 1, 1)
		self.device = device
		self.file_offset = file_offset

	# Pytorch Dataset functions
	def __len__(self):
		return self.num_samples
 
	def __getitem__(self, index):
		# HACK FOR 1 INDEXED DATASETS
		data = torch.tensor(self.read_and_bin_input_file(index + self.file_offset), device=self.device)
		return (data, self.training_samples[index,:,:,:,:], self.labels[index])
		
	def read_labels_file(self, file):
		# Open CSV file that describes our samples
		des_spikes = []
		labels = []
		with open(file, 'r') as testing_file:
			reader = csv.reader(testing_file, delimiter='\t')
			# Skip header
			next(reader, None)
			for line in reader:
				# Append num_classes values with negative class number of target spikes
				ext_list = [self.net_params['negative_spikes']] * self.net_params['num_classes']
				# Assign positive spikes to correct class
				ext_list[int(line[1])] = self.net_params['positive_spikes']
				des_spikes.extend(ext_list)
				labels.append(int(line[1]))
		return des_spikes, int(len(des_spikes) / self.net_params['num_classes']), labels

	def process_event(self, raw_bytes):
		# Ts is the last 23 bits of the raw_bytes array
		ts = ((raw_bytes[2] << 16) | (raw_bytes[3] << 8) | raw_bytes[4]) & 0x7FFFFF
		return (raw_bytes[0], raw_bytes[1], raw_bytes[2] >> 7, ts)

	# TODO optimize, remove iteration
	# TODO make generic to 1d and 2d spike files
	def read_and_bin_input_file(self, index):
		file_name = self.dataset_path + str(index) + ".bs2"
		spike = 1.0 / self.net_params['t_s']
		time_scaling = 1.0 / (self.net_params['time_unit'] * self.net_params['t_s'])
		n_timesteps = int((self.net_params['t_end'] - self.net_params['t_start']) / self.net_params['t_s'])
		# Preallocate numpy array
		binned_array = np.zeros((self.net_params['input_channels'], self.net_params['input_y'], self.net_params['input_x'], n_timesteps), dtype=np.float32)
		with open(file_name, 'rb') as input_file:
			for raw_spike in iter(lambda: input_file.read(self.EVENT_BIN_SIZE), b''):
				(ev_x, ev_y, ev_p, ev_ts) = self.process_event(raw_spike)
				time_position = int(ev_ts * time_scaling)
				# TODO do truncation if ts over t_end, checks on x and y
				binned_array[ev_p, ev_y, ev_x, time_position] = spike
		return binned_array