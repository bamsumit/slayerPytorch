from .slayer import spikeLayer as layer
from .slayerLoihi import spikeLayer as loihi
# from slayer import yamlParams as params
from .slayerParams import yamlParams as params
from .spikeLoss import spikeLoss as loss
from .spikeClassifier import spikeClassifier as predict
from . import spikeFileIO as io
from . import utils

'''
This modules bundles various SLAYER PyTorch modules as a single package.
The complete module can be imported as
>>> import slayerSNN as snn
* The spikeLayer will be available as snn.layer
* The SLAYER Loihi layer will be available as snn.loihi
* The yaml-parameter reader will be availabe as snn.params
* The spike-loss module will be available as snn.loss
* The spike-classifier module will be available as snn.predict
* The spike-IO module will be available as snn.io
* The quantize module will be available as snn.quantize 
'''