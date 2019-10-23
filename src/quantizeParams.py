import torch

class quantizeWeights(torch.autograd.Function):
    '''
    This class provides routine to quantize the weights during forward propagation pipeline.
    The backward propagation pipeline passes the gradient as it it, without any modification.

    Arguments;
        * ``weights``: full precision weight tensor.
        * ``step``: quantization step size. Default: 1

    Usage:

    >>> # Quantize weights in step of 0.5
    >>> stepWeights = quantizeWeights.apply(fullWeights, 0.5)
    '''
    @staticmethod
    def forward(ctx, weights, step=1):
        '''
        '''
        # return weights
        # print('Weights qunatized with step', step)
        return torch.round(weights / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        return gradOutput, None

def quantize(weights, step=1):
    '''
    This function provides a wrapper around quantizeWeights.

    Arguments;
        * ``weights``: full precision weight tensor.
        * ``step``: quantization step size. Default: 1

    Usage:

    >>> # Quantize weights in step of 0.5
    >>> stepWeights = quantize(fullWeights, step=0.5)
    '''
    return quantizeWeights.apply(weights, step)
