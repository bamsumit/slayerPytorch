from numpy.lib.arraysetops import isin
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
    def __init__(self, parameter_file_path=None, dict=None):
        if dict is None:
            with open(parameter_file_path, 'r') as param_file:
                self.parameters = yaml.safe_load(param_file)
        else:
            self.parameters = dict

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)

    def print(self, key=None):
        if key is None:
            printConfig(self.parameters)
        else:
            print(key + ':')
            printConfig(self.parameters[key], pre='    ')

def printConfig(obj, pre=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                print(pre + key + ' :')
                printConfig(value, pre=pre+'    ')
            else:
                print(pre + '{:10s} : {}'.format(str(key), value))
    elif isinstance(obj, list):
        for l in obj:
            printConfig(pre + '- {}'.format(l))
    else:
        print(pre + '{}'.format(obj))
