from .slayer import spikeLayer as layer
from .slayerLoihi import spikeLayer as loihi
# from slayer import yamlParams as params
from .slayerParams import yamlParams as params
from .spikeLoss import spikeLoss as loss
from .spikeClassifier import spikeClassifier as predict
from . import spikeFileIO as io
from . import utils
# This will be removed later. Kept for compatibility only
from .quantizeParams import quantizeWeights as quantize

# from .slayer import spikeLayer as layer
# from .slayerLoihi import spikeLayer as loihi
# # from slayer import yamlParams as params
# from .slayerParams import yamlParams as params
# from .spikeLoss import spikeLoss as loss
# from .spikeClassifier import spikeClassifier as predict
# from . import spikeFileIO as io
# from .quantizeParams import quantizeWeights as quantize
