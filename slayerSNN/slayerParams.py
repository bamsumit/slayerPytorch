import yaml

# Consider dictionary for easier iteration and better scalability
class yamlParams(object):
	'''
	This class reads yaml parameter file and allows dictionary like access to the members.
	
	Usage:

	.. code-block:: python
		
		import slayerSNN as snn
		netParams = snn.params('path_to_yaml_file')	# OR
		netParams = yamlParams('path_to_yaml_file')

		netParams['training']['learning']['etaW'] = 0.01
		print('Simulation step size        ', netParams['simulation']['Ts'])
		print('Spiking neuron time constant', netParams['neuron']['tauSr'])
		print('Spiking neuron threshold    ', netParams['neuron']['theta'])

		netParams.save('filename.yaml')
	'''
	def __init__(self, parameter_file_path):
		with open(parameter_file_path, 'r') as param_file:
			self.parameters = yaml.safe_load(param_file)

	# Allow dictionary like access
	def __getitem__(self, key):
		return self.parameters[key]

	def __setitem__(self, key, value):
		self.parameters[key] = value

	def save(self, filename):
		with open(filename, 'w') as f:
			yaml.dump(self.parameters, f)