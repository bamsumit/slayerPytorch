import os
import struct
import numpy as np

np_event_type = [('x', np.uint16), ('y', np.uint16), ('p', np.uint8), ('ts', np.uint32)]

class DataReader(object):

	def __init__(self, input_path):
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

	# TO OPTIMIZE, remove iteration, preallocate?
	def read_file(self, file):
		# Preallocate numpy array
		file_size = os.stat(file).st_size
		events = np.ndarray((int(file_size / 5)), dtype=np_event_type)
		with open(file, 'rb') as input_file:
			for (index, raw_spike) in enumerate(iter(lambda: input_file.read(5), b'')):
				events[index] = self.process_event(raw_spike)
		return events