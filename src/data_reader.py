import os
import struct
import numpy as np

np_event_type = [('x', np.uint16), ('y', np.uint16), ('p', np.uint8), ('ts', np.uint32)]

# Consider dictionary for easier iteration and better scalability
class SlayerParams(object):

	def __init__(self):
		self.t_start = None
		self.t_end = None
		self.t_res = None
		self.time_unit = None
		self.input_x = None
		self.input_y = None
		self.input_channels = None

	def is_valid(self):
		# Could do more checks here (positive, t_res < t_end - t_start)
		return (self.t_start != None and self.t_end != None and self.t_res != None and self.time_unit != None and
			self.input_x != None and self.input_y != None and self.input_channels != None)

class DataReader(object):

	# TODO pass parameter structure instead of these numbers
	def __init__(self, input_path, net_params):
		self.EVENT_BIN_SIZE = 5
		if not net_params.is_valid():
			raise ValueError("Network parameters are not valid")
		self.net_params = net_params
		# Get files in folder
		self.input_files = self.get_input_filenames(input_path)
		
	def get_input_filenames(self, path):
		# Preprocess path to add trailing slash
		path = path if path.endswith('/') else path + '/'
		files = os.listdir(path)
		return [path + f for f in files if f.endswith('.bs2')]

	def process_event(self, raw_bytes):
		ts = int.from_bytes(raw_bytes[2:], byteorder='big') & 0x7FFFFF
		return (raw_bytes[0], raw_bytes[1], raw_bytes[2] >> 7, ts)

	# TO OPTIMIZE, remove iteration
	def read_file(self, file):
		# Preallocate numpy array
		file_size = os.stat(file).st_size
		events = np.ndarray((int(file_size / self.EVENT_BIN_SIZE)), dtype=np_event_type)
		with open(file, 'rb') as input_file:
			for (index, raw_spike) in enumerate(iter(lambda: input_file.read(self.EVENT_BIN_SIZE), b'')):
				events[index] = self.process_event(raw_spike)
		return events

	# NOTE! Matlab version loads positive spikes first, Python version loads negative spikes first
	def bin_spikes(self, raw_spike_array):
		n_inputs = self.net_params.input_x * self.net_params.input_y * self.net_params.input_channels
		n_timesteps = int((self.net_params.t_end - self.net_params.t_start) / self.net_params.t_res)
		binned_array = np.zeros((n_inputs, n_timesteps), dtype=np.uint8)
		# print(binned_array.shape)
		# Now do the actual binning
		for ev in raw_spike_array:
			# TODO cleanup, access by name (ts) not index
			ev_x = ev[0]
			ev_y = ev[1]
			ev_p = ev[2]
			ev_ts = ev[3]
			time_position = int(ev_ts / self.net_params.time_unit)
			# TODO do truncation if ts over t_end, checks on x and y
			input_position = ev_p * (self.net_params.input_x * self.net_params.input_y) + (ev_y * self.net_params.input_x) + ev_x
			binned_array[input_position, time_position] = 1
		return binned_array