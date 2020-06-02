import sys, os

CURRENT_SRC_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_SRC_DIR + "/../../slayerPyTorch/src")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import slayer
import slayerCuda
import slayerLoihiCuda
from .quantizeParams import quantizeWeights

class spikeLayer(slayer.spikeLayer):
    '''
    This class defines the main engine of SLAYER Loihi module.
    It is derived from ``slayer.spikeLayer`` with Loihi specific implementation for
    neuron model, weight quantization.
    All of the routines available for ``slayer.spikeLayer`` are applicable.

    Arguments:
        * ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
            .. code-block:: python

                neuron:
                    type:     LOIHI # neuron type
                    vThMant:  80    # neuron threshold mantessa
                    vDecay:   128   # compartment voltage decay
                    iDecay:   1024  # compartment current decay
                    refDelay: 1     # refractory delay
                    wgtExp:   0     # weight exponent
                    tauRho:   1     # spike function derivative time constant (relative to theta)
                    scaleRho: 1     # spike function derivative scale factor
        * ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
            .. code-block:: python

                simulation:
                    Ts: 1.0         # sampling time (ms)
                    tSample: 300    # time length of sample (ms)

    Usage:

    >>> snnLayer = slayerLoihi.spikeLayer(neuronDesc, simulationDesc)
    '''
    def __init__(self, neuronDesc, simulationDesc):
        if neuronDesc['type'] == 'LOIHI':
            neuronDesc['theta'] = neuronDesc['vThMant'] * 2**6
        
        super(spikeLayer, self).__init__(neuronDesc, simulationDesc)

        self.maxPspKernel = torch.max(self.srmKernel).cpu().data.item()
        print('Max PSP kernel:', self.maxPspKernel)
        print('Scaling neuron[scaleRho] by Max PSP Kernel @slayerLoihi')
        neuronDesc['scaleRho'] /= self.maxPspKernel
        
    def calculateSrmKernel(self):
        srmKernel = self._calculateLoihiPSP()
        return torch.tensor(srmKernel)
        
    def calculateRefKernel(self, SCALE=1000):
        refKernel = self._calculateLoihiRefKernel(SCALE)
        return torch.tensor(refKernel)
        
    def _calculateLoihiPSP(self):
        # u = [0]
        # v = [0]
        u = []
        v = []
        u.append( 1 << (6 + self.neuron['wgtExp'] + 1) ) # +1 to compensate for weight resolution of 2 for mixed synapse mode
        v.append( u[-1] ) # we do not consider bias in slayer
        while v[-1] > 0:
            uNext = ( ( u[-1] * ( (1<<12) - self.neuron['iDecay']) ) >> 12 )
            vNext = ( ( v[-1] * ( (1<<12) - self.neuron['vDecay']) ) >> 12 ) + uNext # again, we do not consider bias in slayer
            u.append(uNext)
            v.append(vNext)

        return  [float(x)/2 for x in v] # scale by half to compensate for 1 in the initial weight

    def _calculateLoihiRefKernel(self, SCALE=1000):
        absoluteRefKernel = np.ones(self.neuron['refDelay']) * (-SCALE * self.neuron['theta'])
        absoluteRefKernel[0] = 0
        relativeRefKernel = [ self.neuron['theta'] ]
        while relativeRefKernel[-1] > 0:
            nextRefKernel = ( relativeRefKernel[-1] * ( (1<<12) - self.neuron['vDecay']) ) >> 12 
            relativeRefKernel.append(nextRefKernel)
        refKernel = np.concatenate( (absoluteRefKernel, -2 * np.array(relativeRefKernel) ) ).astype('float32')
        return refKernel

    def spikeLoihi(self, weightedSpikes):
        '''
        Applies Loihi neuron dynamics to weighted spike inputs and returns output spike tensor.
        The output tensor dimension is same as input.

        NOTE: This function is different than the default ``spike`` function which takes membrane potential (weighted spikes with psp filter applied).
        Since the dynamics is modeled internally, it just takes in weightedSpikes (NOT FILTERED WITH PSP) for accurate Loihi neuron simulation.

        Arguments:
            * ``weightedSpikes``: input spikes weighted by their corresponding synaptic weights.

        Usage:

        >>> outSpike = snnLayer.spikeLoihi(weightedSpikes)
        '''
        return _spike.apply(weightedSpikes, self.srmKernel, self.neuron, self.simulation['Ts'])

    def spikeLoihiFull(self, weightedSpikes):
        '''
        Applies Loihi neuron dynamics to weighted spike inputs and returns output spike, voltage and current.
        The output tensor dimension is same as input.

        NOTE: This function does not have autograd routine in the computational graph.

        Arguments:
            * ``weightedSpikes``: input spikes weighted by their corresponding synaptic weights.

        Usage:

        >>> outSpike, outVoltage, outCurrent = snnLayer.spikeLoihiFull(weightedSpikes)
        '''
        return _spike.loihi(weightedSpikes, self.neuron, self.simulation['Ts'])

    def dense(self, inFeatures, outFeatures, weightScale=100, quantize=True):
        '''
        This function behaves similar to :meth:`slayer.spikeLayer.dense`. 
        The only difference is that the weights are qunatized with step of 2 (as is the case for signed weights in Loihi).
        One can, however, skip the quantization step altogether as well.

        Arguments:
            The arguments that are different from :meth:`slayer.spikeLayer.dense` are listed.
            * ``weightScale``: sale factor of default initialized weights. Default: 100
            * ``qunatize`` (``bool``): flag to quatize the weights or not. Default: True

        Usage:
            Same as :meth:`slayer.spikeLayer.dense`
        '''
        return _denseLayer(inFeatures, outFeatures, weightScale, quantize)

    def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, quantize=True):
        '''
        This function behaves similar to :meth:`slayer.spikeLayer.conv`. 
        The only difference is that the weights are qunatized with step of 2 (as is the case for signed weights in Loihi).
        One can, however, skip the quantization step altogether as well.

        Arguments:
            The arguments that are different from :meth:`slayer.spikeLayer.conv` are listed.
            * ``weightScale``: sale factor of default initialized weights. Default: 100
            * ``qunatize`` (``bool``): flag to quatize the weights or not. Default: True

        Usage:
            Same as :meth:`slayer.spikeLayer.conv`
        '''
        return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, quantize)
    
    def pool(self, kernelSize, stride=None, padding=0, dilation=1):
        '''
        This function behaves similar to :meth:`slayer.spikeLayer.pool`. 
        The only difference is that the weights are qunatized with step of 2 (as is the case for signed weights in Loihi).
        One can, however, skip the quantization step altogether as well.

        Arguments:
            The arguments set is same as :meth:`slayer.spikeLayer.pool`.

        Usage:
            Same as :meth:`slayer.spikeLayer.pool`
        '''
        requiredWeight = quantizeWeights.apply(torch.tensor(1.1 * self.neuron['theta'] / self.maxPspKernel), 2).cpu().data.item()
        # print('Required pool layer weight =', requiredWeight)
        return slayer._poolLayer(requiredWeight/ 1.1, # to compensate for maxPsp
                          kernelSize, stride, padding, dilation)

    def getVoltage(self, membranePotential):
        Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
        voltage = membranePotential.reshape((-1, Ns)).cpu().data.numpy()
        return np.where(voltage <= -500*self.neuron['theta'], self.neuron['theta'] + 1, voltage)

class _denseLayer(slayer._denseLayer):
    def __init__(self, inFeatures, outFeatures, weightScale=1, quantize=True):
        self.quantize = quantize
        super(_denseLayer, self).__init__(inFeatures, outFeatures, weightScale)

    def forward(self, input):
        if self.quantize is True:
            return F.conv3d(input, 
                            quantizeWeights.apply(self.weight, 2), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

class _convLayer(slayer._convLayer):
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, quantize=True):
        self.quantize = quantize
        super(_convLayer, self).__init__(inFeatures, outFeatures, kernelSize, stride, padding, dilation, groups, weightScale)
 
    def forward(self, input):
        if self.quantize is True:
            return F.conv3d(input, 
                            quantizeWeights.apply(self.weight, 2), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)


class _spike(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def loihi(weightedSpikes, neuron, Ts):
        iDecay = neuron['iDecay']
        vDecay = neuron['vDecay']
        theta  = neuron['theta']
        # wScale = 1 << (6 + neuron['wgtExp'])
        wgtExp = neuron['wgtExp']

        if weightedSpikes.dtype == torch.int32:
            Ts = 1
        
        spike, voltage, current = slayerLoihiCuda.getSpikes((weightedSpikes * Ts).contiguous(), wgtExp, theta, iDecay, vDecay)

        return spike/Ts, voltage, current

    @staticmethod
    def forward(ctx, weightedSpikes, srmKernel, neuron, Ts):
        device = weightedSpikes.device
        dtype  = weightedSpikes.dtype
        pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
        pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
        Ts              = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        srmKernel       = torch.autograd.Variable(srmKernel.clone().detach(), requires_grad=False)

        
        spike, voltage, current = _spike.loihi(weightedSpikes, neuron, Ts)

        ctx.save_for_backward(voltage, threshold, pdfTimeConstant, pdfScale, srmKernel, Ts)
        return spike

    @staticmethod
    def backward(ctx, gradOutput):
        (membranePotential, threshold, pdfTimeConstant, pdfScale, srmKernel, Ts) = ctx.saved_tensors
        spikePdf = pdfScale / pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)

        return slayerCuda.corr(gradOutput * spikePdf, srmKernel, Ts), None, None, None
