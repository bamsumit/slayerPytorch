import torch
from torch.utils.data import Dataset
from ..spikeClassifier import spikeClassifier as predict
import slayerCuda
from datetime import datetime

class Assistant:
    '''
    This class provides standard assistant functionalities for traiing and testing workflow.
    If you want a different workflow than what is available, you should inherit this module and
    overload the particular module to your need.

    Arguments:
        * ``net``: the SLAYER network to be run.
        * ``trainLoader``: training dataloader.
        * ``testLoader``: testing dataloader.
        * ``error``: a function object or a lamda function that takes (output, target, label) as its input and returns
            a scalar error value.
        * ``optimizer``: the learning optimizer.
        * ``scheduler``: the learning scheduler. Default: ``None`` meaning no scheduler will be used.
        * ``stats``: the SLAYER learning stats logger: ``slayerSNN.stats``. Default: ``None`` meaning no stats will be used.
        * ``dataParallel``: flag if dataParallel execution needs to be handled. Default: ``False``.
        * ``showTimeSteps``: flag to print timesteps of the sample or not. Default: ``False``.
        * ``lossScale``: a scale factor to be used while printing the loss. Default: ``None`` meaning no scaling is done.
        * ``printInterval``: number of epochs to print the lerning output once. Default: 1.
    
    Usage:

    .. code-block:: python
        
        assist = assistant(net, trainLoader, testLoader, lambda o, t, l: error.numSpikes(o, t), optimizer, stats)

        for epoch in range(maxEpoch): 
            assist.train(epoch)
            assist.test(epoch)
    '''
    def __init__(self, net, trainLoader, testLoader, error, optimizer, scheduler=None, stats=None, 
                 dataParallel=False, showTimeSteps=False, lossScale=None, printInterval=1):
        self.net = net
        self.module = net.module if dataParallel is True else net
        self.error = error
        self.device = self.module.slayer.srmKernel.device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stats = stats
        self.showTimeSteps = showTimeSteps
        self.lossScale = lossScale
        self.printInterval = printInterval

        self.trainLoader = trainLoader
        self.testLoader = testLoader

    def train(self, epoch=0, breakIter = None, printLog=True):
        '''
        Training assistant fucntion.

        Arguments:
            * ``epoch``: training epoch number.
            * ``breakIter``: number of samples to wait before breaking out of the training loop. 
                ``None`` means go over the complete training samples. Default: ``None``.
        '''
        tSt = datetime.now()
        for i, (input, target, label) in enumerate(self.trainLoader, 0):
            self.net.train()

            input  = input.to(self.device)
            target = target.to(self.device) 

            count = 0
            if self.module.countLog is True:
                output, count = self.net.forward(input)
            else:
                output = self.net.forward(input)
            
            if self.stats is not None:
                self.stats.training.correctSamples += torch.sum( predict.getClass(output) == label ).data.item()
                self.stats.training.numSamples     += len(label)

            loss = self.error(output, target, label)
            if self.stats is not None:
                self.stats.training.lossSum += loss.cpu().data.item() * (1 if self.lossScale is None else self.lossScale)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.module.clamp()

            if self.stats is not None and i%self.printInterval == 0 and printLog is True:
                headerList = ['[{}/{} ({:.0f}%)]'.format(i*self.trainLoader.batch_size, len(self.trainLoader.dataset), 100.0*i/len(self.trainLoader))]
                if self.module.countLog is True:
                    headerList.append('Spike count: ' + ', '.join(['{}'.format(int(c)) for c in torch.sum(count, dim=0).tolist()]))
                if self.showTimeSteps is True:
                    headerList.append('nTimeBins: {}'.format(input.shape[-1]))

                self.stats.print(
                    epoch, i, 
                    (datetime.now() - tSt).total_seconds() / (i+1) / self.trainLoader.batch_size,
                    header= headerList,
                )

            if breakIter is not None and i >= breakIter:
                break

            if self.scheduler is not None:
                self.scheduler.step()

    def test(self, epoch=0, evalLoss=True, slidingWindow=None, breakIter = None, printLog=True):
        '''
        Testing assistant fucntion.

        Arguments:
            * ``epoch``: training epoch number.
            * ``evalLoss``: a flag to enable or disable loss evalutaion. Default: ``True``.
            * ``slidingWindow``: the length of sliding window to use for continuous output prediction over time. 
                ``None`` means total spike count is used to produce one output per sample. If it is not
                ``None``, ``evalLoss`` is overwritten to ``False``. Default: ``None``.
            * ``breakIter``: number of samples to wait before breaking out of the testing loop. 
                ``None`` means go over the complete training samples. Default: ``None``.
        '''
        if slidingWindow is not None:
            filter = torch.ones((slidingWindow)).to(self.device)
            evalLoss = False

        tSt = datetime.now()
        for i, (input, target, label) in enumerate(self.testLoader, 0):
            self.net.eval()

            with torch.no_grad():
                input  = input.to(self.device)
                target = target.to(self.device) 

                count = 0
                if self.module.countLog is True:
                    output, count = self.net.forward(input)
                else:
                    output = self.net.forward(input)

                if slidingWindow is None:
                    if self.stats is not None:
                        self.stats.testing.correctSamples += torch.sum( predict.getClass(output) == label ).data.item()
                        self.stats.testing.numSamples     += len(label)
                else:
                    filteredOutput = slayerCuda.conv(output.contiguous(), filter, 1)[..., slidingWindow:]
                    predictions = torch.argmax(filteredOutput.reshape(-1, filteredOutput.shape[-1]), dim=0)
                    
                    # print(output.shape, predictions.shape)
                    # print(predictions[:100])
                    # print(label)
                    # print(torch.sum(predictions == label).item())
                    # print(torch.sum(predictions == label).item() / predictions.shape[0])

                    # assert False, 'Just braking'
                    
                    if self.stats is not None:
                        self.stats.testing.correctSamples += torch.sum(predictions == label.to(self.device)).item()
                        self.stats.testing.numSamples     += predictions.shape[0]

                if evalLoss is True:
                    loss = self.error(output, target, label)
                    if self.stats is not None:
                        self.stats.testing.lossSum += loss.cpu().data.item() * (1 if self.lossScale is None else self.lossScale)
                else:
                    if self.stats is not None:
                        if slidingWindow is None:
                            self.stats.testing.lossSum += (1 if self.lossScale is None else self.lossScale)
                        else:
                            self.stats.testing.lossSum += predictions.shape[0] * (1 if self.lossScale is None else self.lossScale)

            if self.stats is not None and i%self.printInterval == 0 and printLog is True:
                headerList = ['[{}/{} ({:.0f}%)]'.format(i*self.testLoader.batch_size, len(self.testLoader.dataset), 100.0*i/len(self.testLoader))]
                if self.module.countLog is True:
                    headerList.append('Spike count: ' + ', '.join(['{}'.format(int(c)) for c in torch.sum(count, dim=0).tolist()]))
                if self.showTimeSteps is True:
                    headerList.append('nTimeBins: {}'.format(input.shape[-1]))

                self.stats.print(
                    epoch, i, 
                    (datetime.now() - tSt).total_seconds() / (i+1) / self.testLoader.batch_size,
                    header= headerList,
                )

            if breakIter is not None and i >= breakIter:
                break
