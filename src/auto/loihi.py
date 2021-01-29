# Sumit Bam Shrestha 09/28/2020 5pm
# =================================
# This is a wrapper code that generates feedforward slayerSNN network from a 
# network config file (*.yaml). This will also include modules to output a 
# network description file (*.hdf5) OR SOME OTHER FORMAT (TO BE DECIDED) which
# will be directly loadable in nxsdk (PERHAPS THIS NEEDS REMOVING LATER) 
# module to load the trained network in Loihi hardware.
#
# This module should be merged with slayerSNN.loihi later and served from 
# SLAYER-PyTorch module
# It shall be accessible as slayerSNN.auto.loihi

from .. import utils
from ..slayerLoihi import spikeLayer as loihi

from collections import _count_elements
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import re
import h5py

class denseBlock(torch.nn.Module):
    '''
    This class creates a dense layer block with Loihi neuron. It groups the 
    synaptic interaction, Loihi neuron response and the associated delays.

    Arguments:
        * ``slayer`` (``slayerLoihi.slayer``): pre-initialized slayer loihi module.
        * ``inFeatures``: number of input features.
        * ``outFeatures``: number of output features.
        * ``weightScale``: scale factor of the defaule initialized weights. Default: 100
        * ``preHoodFx``: a function that operates on weight before applying it. Could be used for quantization etc.
        * ``weightNorm``: a flag to indicate if weight normalization should be applied or not. Default: False
        * ``delay``: a flag to inidicate if axonal delays should be applied or not. Default: False
        * ``maxDelay``: maximum allowable delay. Default: 62
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = denseBlock(self.slayer, 512, 10)
    '''
    def __init__(self, slayer, inFeatures, outFeatures, weightScale=100, 
                 preHookFx = lambda x: utils.quantize(x, step=2), weightNorm=False, 
                 delay=False, maxDelay=62, countLog=False):
        super(denseBlock, self).__init__()
        self.slayer = slayer
        self.weightNorm = weightNorm
        if weightNorm is True:
            self.weightOp = torch.nn.utils.weight_norm(slayer.dense(inFeatures, outFeatures, weightScale, preHookFx), name='weight')
        else:
            self.weightOp = slayer.dense(inFeatures, outFeatures, weightScale, preHookFx)
        self.delayOp  = slayer.delay(outFeatures) if delay is True else None
        self.countLog = countLog
        self.gradLog = True
        self.maxDelay = maxDelay

        self.paramsDict = {
            'inFeatures'  : inFeatures,
            'outFeatures' : outFeatures,
        } 

    def forward(self, spike):
        spike = self.slayer.spikeLoihi(self.weightOp(spike))
        spike = self.slayer.delayShift(spike, 1)
        if self.delayOp is not None:
            spike = self.delayOp(spike)

        if self.countLog is True:
            return spike, torch.sum(spike)
        else:
            return spike

class convBlock(torch.nn.Module):
    '''
    This class creates a conv layer block with Loihi neuron. It groups the 
    synaptic interaction, Loihi neuron response and the associated delays.

    Arguments:
        * ``slayer`` (``slayerLoihi.slayer``): pre-initialized slayer loihi module.
        * ``inChannels``: number of input channels.
        * ``outChannels``: number of output channels.
        * ``kernelSize``: size of convolution kernel.
        * ``stride``: size of convolution stride. Default: 1
        * ``padding``: size of padding. Default: 0
        * ``dialtion``: size of convolution dilation. Default: 1
        * ``groups``: number of convolution groups. Default: 1
        * ``weightScale``: scale factor of the defaule initialized weights. Default: 100
        * ``preHoodFx``: a function that operates on weight before applying it. Could be used for quantization etc.
            Default: quantization in step of 2 (Mixed weight mode in Loihi)
        * ``weightNorm``: a flag to indicate if weight normalization should be applied or not. Default: False
        * ``delay``: a flag to inidicate if axonal delays should be applied or not. Default: False
        * ``maxDelay``: maximum allowable delay. Default: 62
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = convBlock(self.slayer, 16, 31, 3, padding=1)
        spike = blk(spike)
    '''
    def __init__(self, slayer, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, 
                 preHookFx = lambda x: utils.quantize(x, step=2), weightNorm=False, 
                 delay=False, maxDelay=62, countLog=False):
        super(convBlock, self).__init__()
        self.slayer = slayer
        self.weightNorm = weightNorm
        if weightNorm is True:
            self.weightOp = torch.nn.utils.weight_norm(
                slayer.conv(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx), 
                name='weight',
            )
        else:
            self.weightOp = slayer.conv(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx)
        # only channel wise delay is supported for conv layer
        # for neuron wise delay, one will need to write a custom block as it would require the spatial dimension as well
        self.delayOp  = slayer.delay(outChannels) if delay is True else None    
        self.countLog = countLog
        self.gradLog = True
        self.maxDelay = maxDelay

        self.paramsDict = {
            'inChannels'  : inChannels,
            'outChannels' : outChannels,
            'kernelSize'  : kernelSize,
            'stride'      : stride,
            'padding'     : padding,
            'dilation'    : dilation,
            'groups'      : groups,
        } 

    def forward(self, spike):
        spike = self.slayer.spikeLoihi(self.weightOp(spike))
        spike = self.slayer.delayShift(spike, 1)
        if self.delayOp is not None:
            spike = self.delayOp(spike)

        if self.countLog is True:
            return spike, torch.sum(spike)
        else:
            return spike

class poolBlock(torch.nn.Module):
    '''
    This class creates a pool layer block with Loihi neuron. It groups the 
    synaptic interaction, Loihi neuron response and the associated delays.

    Arguments:
        * ``slayer`` (``slayerLoihi.slayer``): pre-initialized slayer loihi module.
        * ``kernelSize``: size of pooling kernel.
        * ``stride``: size of pooling stride. Default: None(same as ``kernelSize``)
        * ``padding``: size of padding. Default: 0
        * ``dialtion``: size of convolution dilation. Default: 1
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = poolBlock(self.slayer, 2)
        spike = blk(spike)
    '''
    def __init__(self, slayer, kernelSize, stride=None, padding=0, dilation=1, countLog=False):
        super(poolBlock, self).__init__()
        self.slayer = slayer
        self.weightOp = slayer.pool(kernelSize, stride, padding, dilation)
        self.countLog = countLog
        self.delayOp = None # it does not make sense to have axonal delays after pool block
        self.gradLog = False # no need to monitor gradients
        self.paramsDict = {
            'kernelSize' : kernelSize,
            'stride'     : kernelSize if stride is None else stride,
            'padding'    : padding,
            'dilation'   : dilation, 
        }

    def forward(self, spike):
        spike = self.slayer.spikeLoihi(self.weightOp(spike))
        spike = self.slayer.delayShift(spike, 1)

        if self.countLog is True:
            return spike, None  # return None for count. It does not make sense to count for pool layer
        else:
            return spike

class flattenBlock(torch.nn.Module):
    '''
    This class flattens the spatial dimension. The resulting tensor is compatible with dense layer.

    Arguments:
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = flattenBlock(self.slayer, True)
        spike = blk(spike)
    '''
    def __init__(self, countLog=False):
        super(flattenBlock, self).__init__()
        self.delayOp = None
        self.weightOp = None
        self.gradLog = False
        self.countLog = countLog
        self.paramsDict = {}

    def forward(self, spike):
        if self.countLog is True:
            return spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1])), None
        else:
            return spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))

class averageBlock(torch.nn.Module):
    '''
    This class averages the spikes among n different output groups for population voting.

    Arguments:
        * ``nOutputs``: number of output groups (Equal to the number of ouptut classes).
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = averageBlock(self.slayer, nOutputs=10)
        spike = blk(spike)
    '''
    def __init__(self, nOutputs, countLog=False):
        super(averageBlock, self).__init__()
        self.nOutputs = nOutputs
        self.delayOp = None
        self.weightOp = None
        self.gradLog = False
        self.countLog = countLog
        self.paramsDict = {}

    def forward(self, spike):
        N, _, _, _, T = spike.shape
        if self.countLog is True:
            return torch.mean(spike.reshape((N, self.nOutputs, -1, 1, T)), dim=2, keepdim=True), None
        else:
            return torch.mean(spike.reshape((N, self.nOutputs, -1, 1, T)), dim=2, keepdim=True)




class Network(torch.nn.Module):
    '''
    This class encapsulates the network creation from the networks described in netParams
    configuration. A netParams configuration is ``slayerSNN.slayerParams.yamlParams`` which
    can be initialized from a yaml config file or a dictionary.

    In addition to the standard network ``forward`` function, it also includes ``clamp`` function 
    for clamping delays, ``gradFlow`` function for monitioring the gradient flow, and ``genModel``
    function for exporting a hdf5 file which is a packs network specification and trained 
    parameter into a single file that can be possibly used to generate the inference network 
    specific to a hardware, with some support.

    Arguments:
        * ``nOutputs``: number of output groups (Equal to the number of ouptut classes).
        * ``countLog``: a flag to indicate if a log of spike count should be maintained and passed around or not. 
            Default: False
    
    Usage:

    .. code-block:: python
        
        blk = averageBlock(self.slayer, nOutputs=10)
        spike = blk(spike)
    '''
    def __init__(self, netParams, preHookFx=lambda x: utils.quantize(x, step=2), weightNorm=False, countLog=False):
        super(Network, self).__init__()

        self.netParams = netParams
        self.netParams.print('simulation')
        print('')
        self.netParams.print('neuron')
        print('')
        # TODO print netParams

        # initialize slayer
        slayer = loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer

        self.inputShape = None
        self.nOutput = None
        self.weightNorm = weightNorm
        self.preHookFx =preHookFx
        self.countLog = countLog
        self.layerDims = []

        # parse the layer information
        self.blocks = self._parseLayers()
        
        # TODO pass through core usage estimator
        print('TODO core usage estimator')


    def _layerType(self, dim):
        if type(dim) is int:
            return 'dense'
        elif dim.find('c') != -1:
            return 'conv'
        elif dim.find('avg') != -1:
            return 'average'
        elif dim.find('a') != -1:
            return 'pool'
        elif dim.find('x') != -1:
            return 'input'
        else:
            raise Exception('Could not parse the layer description. Found {}'.format(dim))
        # return [int(i) for i in re.findall(r'\d+', dim)]

    def _tableStr(self, typeStr='', width=None, height=None, channel=None, kernel=None, stride=None, 
                 padding=None, delay=False, numParams=None, header=False, footer=False):
        if header is True:
            return '|{:10s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:10s}|'.format(
                    '   Type   ', '  W  ', '  H  ', '  C  ', ' ker ', ' str ', ' pad ', 'delay', '  params  ')
        elif footer is True and numParams is not None:
            return '|{:10s} {:5s} {:5s} {:5s} {:5s} {:5s} {:5s} {:5s}|{:-10d}|'.format(
                    'Total', '', '', '', '', '', '', '', numParams)
        else:
            entry = '|'
            entry += '{:10s}|'.format(typeStr)
            entry += '{:-5d}|'.format(width)
            entry += '{:-5d}|'.format(height)
            entry += '{:-5d}|'.format(channel)
            entry += '{:-5d}|'.format(kernel) if kernel is not None else '{:5s}|'.format('')
            entry += '{:-5d}|'.format(stride) if stride is not None else '{:5s}|'.format('')
            entry += '{:-5d}|'.format(padding) if padding is not None else '{:5s}|'.format('')
            entry += '{:5s}|'.format(str(delay))
            entry += '{:-10d}|'.format(numParams) if numParams is not None else '{:10s}|'.format('')

            return entry

    def _parseLayers(self):
        i = 0
        blocks = torch.nn.ModuleList()
        layerDim = [] # CHW
        is1Dconv = False

        print('\nNetwork Architecture:')
        # print('=====================')
        print(self._tableStr(header=True))

        for layer in self.netParams['layer']:
            layerType = self._layerType(layer['dim'])
            # print(i, layerType)

            # if layer has neuron feild, then use the slayer initialized with it and self.netParams['simulation']
            if 'neuron' in layer.keys():
                print(layerType, 'using individual slayer')
                slayer = loihi(layer['neuron'], self.netParams['simulation'])
            else:
                slayer = self.slayer

            if i==0 and self.inputShape is None: 
                if layerType == 'input':
                    self.inputShape = tuple([int(numStr) for numStr in re.findall(r'\d+', layer['dim'])])
                    if len(self.inputShape) == 3:
                        layerDim = list(self.inputShape)[::-1]
                    elif len(self.inputShape) == 2:
                        layerDim = [1, self.inputShape[1], self.inputShape[0]]
                    else:
                        raise Exception('Could not parse the input dimension. Got {}'.format(self.inputShape))
                elif layerType == 'dense':
                    self.inputShape = tuple([layer['dim']])
                    layerDim = [layer['dim'], 1, 1]
                else:
                    raise Exception('Input dimension could not be determined! It should be the first entry in the' 
                                    + "'layer' feild.")
                # print(self.inputShape)
                print(self._tableStr('Input', layerDim[2], layerDim[1], layerDim[0]))
                if layerDim[1] == 1:
                    is1Dconv = True
            else:
                # print(i, layer['dim'], self._layerType(layer['dim']))
                if layerType == 'conv':
                    params = [int(i) for i in re.findall(r'\d+', layer['dim'])]
                    inChannels  = layerDim[0]
                    outChannels = params[0]
                    kernelSize  = params[1]
                    stride      = layer['stride']   if 'stride'   in layer.keys() else 1
                    padding     = layer['padding']  if 'padding'  in layer.keys() else kernelSize//2
                    dilation    = layer['dilation'] if 'dilation' in layer.keys() else 1
                    groups      = layer['groups']   if 'groups'   in layer.keys() else 1
                    weightScale = layer['wScale']   if 'wScale'   in layer.keys() else 100
                    delay       = layer['delay']    if 'delay'    in layer.keys() else False
                    maxDelay    = layer['maxDelay'] if 'maxDelay' in layer.keys() else 62
                    # print(i, inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale)
                    
                    if is1Dconv is False:
                        blocks.append(convBlock(slayer, inChannels, outChannels, kernelSize, stride, padding, 
                                                dilation, groups, weightScale, self.preHookFx, self.weightNorm, 
                                                delay, maxDelay, self.countLog))
                        layerDim[0] = outChannels
                        layerDim[1] = int(np.floor((layerDim[1] + 2*padding - dilation * (kernelSize - 1) - 1)/stride + 1))
                        layerDim[2] = int(np.floor((layerDim[2] + 2*padding - dilation * (kernelSize - 1) - 1)/stride + 1))
                    else:
                        blocks.append(convBlock(slayer, inChannels, outChannels, [1, kernelSize], [1, stride], [0, padding], 
                                                [1, dilation], groups, weightScale, self.preHookFx, self.weightNorm, 
                                                delay, maxDelay, self.countLog))
                        layerDim[0] = outChannels
                        layerDim[1] = 1
                        layerDim[2] = int(np.floor((layerDim[2] + 2*padding - dilation * (kernelSize - 1) - 1)/stride + 1))
                    self.layerDims.append(layerDim.copy())

                    print(self._tableStr('Conv', layerDim[2], layerDim[1], layerDim[0], kernelSize, stride, padding, 
                          delay, sum(p.numel() for p in blocks[-1].parameters() if p.requires_grad)))
                elif layerType == 'pool':
                    params = [int(i) for i in re.findall(r'\d+', layer['dim'])]
                    # print(params[0])
                    
                    blocks.append(poolBlock(slayer, params[0], countLog=self.countLog))
                    layerDim[1] = int(np.ceil(layerDim[1] / params[0]))
                    layerDim[2] = int(np.ceil(layerDim[2] / params[0]))
                    self.layerDims.append(layerDim.copy())

                    print(self._tableStr('Pool', layerDim[2], layerDim[1], layerDim[0], params[0]))
                elif layerType == 'dense':
                    params = layer['dim']
                    # print(params)
                    if layerDim[1] != 1 or layerDim[2] != 1: # needs flattening of layers
                        blocks.append(flattenBlock(self.countLog ))
                        layerDim[0] = layerDim[0] * layerDim[1] * layerDim[2]
                        layerDim[1] = layerDim[2] = 1
                        self.layerDims.append(layerDim.copy())
                    weightScale = layer['wScale']   if 'wScale'   in layer.keys() else 100
                    delay       = layer['delay']    if 'delay'    in layer.keys() else False
                    maxDelay    = layer['maxDelay'] if 'maxDelay' in layer.keys() else 62
                    
                    blocks.append(denseBlock(slayer, layerDim[0], params, weightScale, self.preHookFx, 
                                  self.weightNorm, delay, maxDelay, self.countLog))
                    layerDim[0] = params
                    layerDim[1] = layerDim[2] = 1
                    self.layerDims.append(layerDim.copy())

                    print(self._tableStr('Dense', layerDim[2], layerDim[1], layerDim[0], delay=delay, 
                                        numParams=sum(p.numel() for p in blocks[-1].parameters() if p.requires_grad)))
                elif layerType == 'average':
                    params = [int(i) for i in re.findall(r'\d+', layer['dim'])]
                    layerDim[0] = params[0]
                    layerDim[1] = layerDim[2] = 1
                    self.layerDims.append(layerDim.copy())

                    blocks.append(averageBlock(nOutputs=layerDim[0], countLog=self.countLog))
                    print(self._tableStr('Average', 1, 1, params[0]))

                i += 1
        self.nOutput = layerDim[0] * layerDim[1] * layerDim[2]
        print(self._tableStr(numParams=sum(p.numel() for p in blocks.parameters() if p.requires_grad), footer=True))
        return blocks

    def forward(self, spike):
        '''
        Forward operation of the network.

        Arguments:
            * ``spike``: Input spke tensor.
        
        Usage:

        .. code-block:: python
            
            net = Network(netParams)
            spikeOut = net.forward(spike)
        '''
        count = []

        for b in self.blocks:
            # print(b)
            # print(b.countLog)
            if self.countLog is True:
                spike, cnt = b(spike)
                if cnt is not None:
                    count.append(cnt.item())
            else:
                spike = b(spike)
            # print(spike.shape)
        
        if self.countLog is True:
            return spike, torch.tensor(count).reshape((1, -1)).to(spike.device)
        else:
            return spike

    def clamp(self):
        '''
        Clamp routine for delay parameters after gradient step to ensure positive value and limit 
        the maximum value.

        Usage:

        .. code-block:: python
            
            net = Network(netParams)
            net.clamp()
        '''
        for d in self.blocks:
            if d.delayOp is not None:
                # d.delayOp.delay.data.clamp_(0, 62)
                d.delayOp.delay.data.clamp_(0, d.maxDelay)
                # print(d.maxDelay)
                # print()
                # print(d.delayOp.delay.shape)

    def gradFlow(self, path):
        '''
        A method to monitor the flow of gradient across the layers. Use it to monitor exploding and
        vanishing gradients. ``scaleRho`` must be tweaked to ensure proper gradient flow. Usually
        monitoring it for first few epochs is good enough.

        Usage:

        .. code-block:: python
            
            net = Network(netParams)
            net.gradFlow(path_to_save)
        '''
        gradNorm = lambda x: torch.norm(x).item()/torch.numel(x)
        grad = []

        for l in self.blocks:
            # print(l)
            if l.gradLog is True:
                if l.weightNorm is True:
                    grad.append(gradNorm(l.weightOp.weight_g.grad))
                else:
                    grad.append(gradNorm(l.weightOp.weight.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

    def genModel(self, fname):
        '''
        This function exports a hdf5 encapsulated neuron parameter, network structure, the weight
        and delay parameters of the trained network. This is intended to be platform indepenent
        representation of the network. The basic protocol of the file is as follows:

        .. code-block::

            |->simulation # simulation description
            |   |->Ts # sampling time. Usually 1
            |   |->tSample # length of the sample to run
            |->layer # description of network layer blocks such as input, dense, conv, pool, flatten, average
                |->0
                |   |->{shape, type, ...} # each layer description has ateast shape and type attribute
                |->1
                |   |->{shape, type, ...}
                :
                |->n
                    |->{shape, type, ...}

            input  : {shape, type}
            flatten: {shape, type}
            average: {shape, type}
            dense  : {shape, type, neuron, inFeatures, outFeatures, weight, delay(if available)}
            pool   : {shape, type, neuron, kernelSize, stride, padding, dilation, weight}
            conv   : {shape, type, neuron, inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weight, delay(if available)}
                                    |-> this is the description of the compartment parameters
                                    |-> {iDecay, vDecay, vThMant, refDelay, ... (other additional parameters can exist)}

        Usage:

        .. code-block:: python
            
            net = Network(netParams)
            net.genModel(path_to_save)
        '''
        qWeights = lambda x: self.preHookFx(x).cpu().data.numpy().squeeze()
        qDelays  = lambda d: torch.floor(d).flatten().cpu().data.numpy().squeeze()

        h = h5py.File(fname, 'w')

        simulation = h.create_group('simulation')

        for key, value in self.netParams['simulation'].items():
            # print(key, value)
            simulation[key] = value
        
        layer = h.create_group('layer')
        layer.create_dataset('0/type', (1, ), 'S10', [b'input'])
        layer.create_dataset('0/shape', data=np.array([self.inputShape[2], self.inputShape[1], self.inputShape[0]]))
        for i, block in enumerate(self.blocks):
            # print(block.__class__.__name__, self.layerDims[i])
            # find the layerType from the block name. Exclude last 5 characters: Block
            layerType = block.__class__.__name__[:-5]
            # print(layerType.encode('ascii', 'ignore'))
            layer.create_dataset('{}/type'.format(i+1), (1, ), 'S10', [layerType.encode('ascii', 'ignore')])
            # print(i, self.layerDims[i])
            layer.create_dataset('{}/shape'.format(i+1), data=np.array(self.layerDims[i]))
            
            if block.weightOp is not None:
                if self.weightNorm is True and layerType != 'pool':
                    torch.nn.utils.remove_weight_norm(block.weightOp, name='weight')
                layer.create_dataset('{}/weight'.format(i+1), data=qWeights(block.weightOp.weight))
            
            if block.delayOp is not None:
                layer.create_dataset('{}/delay'.format(i+1), data=qDelays(block.delayOp.delay))
            
            for key, param in block.paramsDict.items():
                layer.create_dataset('{}/{}'.format(i+1, key), data=param)
            if layerType != 'flatten' and layerType != 'average':
                for key, value in block.slayer.neuron.items():
                    # print(i, key, value)
                    layer.create_dataset('{}/neuron/{}'.format(i+1, key), data=value)

        h.close()


    def loadModel(self, fname):
        '''
        This function loads the network from a perviously saved hdf5 file using ``genModel``.

        Usage:

        .. code-block:: python
            
            net = Network(netParams)
            net.loadModel(path_of_model)
        '''
        # only the layer weights and delays shall be loaded
        h = h5py.File(fname, 'r')

        # one more layer for input layer in the hdf5 file
        assert len(h['layer']) == len(self.blocks) + 1, 'The number of layers in the network does not match with the number of layers in the file {}. Expected {}, found {}'.format(fname, len(self.blocks) + 1, len(h['layer']))

        for i, block in enumerate(self.blocks):
            idxKey = '{}'.format(i+1)
            blockTypeStr = block.__class__.__name__[:-5]
            layerTypeStr = h['layer'][idxKey]['type'][()][0].decode('utf-8')
            assert layerTypeStr == blockTypeStr, 'The layer typestring do not match. Found {} in network and {} in file.'.format(blockTypeStr, layerTypeStr)

            if block.weightOp is not None:
                if self.weightNorm is True and layerTypeStr != 'pool':
                    torch.nn.utils.remove_weight_norm(block.weightOp, name='weight')
                block.weightOp.weight.data = torch.FloatTensor(h['layer'][idxKey]['weight'][()]).reshape(block.weightOp.weight.shape).to(block.weightOp.weight.device)
                if self.weightNorm is True and layerTypeStr != 'pool':
                    block.weightOp = torch.nn.utils.weight_norm(block.weightOp, name='weight')

            if block.delayOp is not None:
                block.delayOp.delay.data = torch.FloatTensor(h['layer'][idxKey]['delay'][()]).reshape(block.delayOp.delay.shape).to(block.delayOp.delay.device)

        

