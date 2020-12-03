import numpy as np
import torch
from torch.utils.data import Dataset
from .. import spikeFileIO as sio

class SlayerDataset(Dataset):
    '''
    This class wraps a basic dataset class to be used in SLAYER training. This allows the use
    of the same basic dataset definition on some other platform other than SLAYER, for e.g. for
    implementation in a neuromorphic hardware with its SDK.

    The basic dataset must return a numpy array of events where each row consists of an AER event
    represented by x, y, polarity and time (in ms).

    Arguments:
        * ``dataset``: basic dataset to be wrapped.
        * ``network``: an ``auto`` module network with which the dataset is intended to be used with. 
            The shape of the tensor is determined from the netowrk definition.
        * ``randomShift``: a flag to indicate if the sample must be randomly shifted in time over the 
            entire sample length. Default: False
        * ``binningMode``: the way the overlapping events are binned. Supports ``SUM`` and ``OR`` binning.
            Default: ``OR``
        * ``fullDataset``: a flag that indicates weather the full dataset is to be processed or not.
            If ``True``, full length of the events is loaded into tensor. This will cause problems with
            default batching, as the number of time bins will not match for all the samples in a minibatch.
            In this case, the dataloader's ``collate_fn`` must be custom defined or a batch size of 1 should
            be used. Default: ``False``
    
    Usage:

    .. code-block:: python
        
        dataset = SlayerDataset(dataset, net)
    '''
    # this expects np event and label from dataset
    # np event should have events ordered in x, y, p, t(ms)
    def __init__(self, dataset, network, randomShift=False, binningMode='OR', fullDataset=False):
        # fullDataset = True superseds randomShift, nTimeBins and tensorShape. It is expected to be run with batch size of 1 only
        super(SlayerDataset, self).__init__()
        self.dataset = dataset
        self.samplingTime = network.netParams['simulation']['Ts']
        self.sampleLength = network.netParams['simulation']['tSample']
        self.nTimeBins    = int(self.sampleLength/self.samplingTime)
        self.inputShape   = network.inputShape
        self.nOutput      = network.nOutput
        self.tensorShape  = (self.inputShape[2], self.inputShape[1], self.inputShape[0], self.nTimeBins)
        self.randomShift  = randomShift
        self.binningMode  = binningMode
        self.fullDataset  = fullDataset

    def __getitem__(self, index):
        event, label = self.dataset[index]

        if self.fullDataset is False:
            inputSpikes = sio.event(
                event[:, 0], event[:, 1], event[:, 2], event[:, 3]
            ).toSpikeTensor(
                torch.zeros(self.tensorShape), 
                samplingTime=self.samplingTime,
                randomShift=self.randomShift,
                binningMode=self.binningMode,
            )
        else:
            nTimeBins = int(np.ceil(event[:, 3].max()))
            tensorShape = (self.inputShape[2], self.inputShape[1], self.inputShape[0], nTimeBins)
            inputSpikes = sio.event(
                event[:, 0], event[:, 1], event[:, 2], event[:, 3]
            ).toSpikeTensor(
                torch.zeros(tensorShape), 
                samplingTime=self.samplingTime,
                randomShift=self.randomShift,
                binningMode=self.binningMode,
            )

        desiredClass = torch.zeros((self.nOutput, 1, 1, 1))
        desiredClass[label, ...] = 1

        return inputSpikes, desiredClass, label

    def __len__(self):
        return len(self.dataset)