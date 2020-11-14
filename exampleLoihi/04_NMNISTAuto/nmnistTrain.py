import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

import zipfile
import numpy as np
if __name__ == '__main__':
    # loading GPU related modules only when executing this script
    import torch
    from torch.utils.data import Dataset, DataLoader
    import slayerSNN as snn
    import slayerSNN.auto # this can be referenced as snn.auto
import zipfile

netDesc = {
    'simulation' : {'Ts': 1, 'tSample': 350},
    'neuron' : {
        'type'     : 'LOIHI',
        'vThMant'  : 80,
        'vDecay'   : 128,
        'iDecay'   : 1024,
        'refDelay' : 1,
        'wgtExp'   : 0,
        'tauRho'   : 1,
        'scaleRho' : 1,
    },
    'layer' : [
        {'dim' : '34x34x2'},
        {'dim' : 512},
        {'dim' : 512},
        {'dim' : 11},
    ],
    'training' : {
        'error' : {
            'type' : 'NumSpikes',
            'tgtSpikeRegion' : {'start': 0, 'stop': 300},
            'tgtSpikeCount' : {True: 60, False: 10},
        }
    }
}

class nmnistDataset():
    def __init__(self, datasetPath, train=True):
        self.path = datasetPath 
        if train is True:
            self.samples = np.loadtxt(datasetPath + '/train1K.txt').astype('int')
        else:
            self.samples = np.loadtxt(datasetPath + '/test100.txt').astype('int')

    def __getitem__(self, index):
        # Read inoput and label
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]
        # Read input spike
        TD = snn.io.read2Dspikes(self.path + '/' + str(inputIndex.item()) + '.bs2')
        event = np.zeros((len(TD.t), 4))
        event[:,0] = TD.x
        event[:,1] = TD.y
        event[:,2] = TD.p
        event[:,3] = TD.t
        
        return event, classLabel
    
    def __len__(self):
        return self.samples.shape[0]

# Extract NMNISTsmall dataset
with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
    for member in zip_file.namelist():
        if not os.path.exists('./' + member):
            zip_file.extract(member, './')

if __name__ == '__main__':
    modelName = 'nmnist'
    trainedFolder = 'Trained'
    os.makedirs(trainedFolder, exist_ok=True)
    device = torch.device('cuda')
    
    # create params object from dictionary / yaml file
    netParams = snn.params(dict=netDesc)
    
    # automatically create the network
    net = snn.auto.loihi.Network(netParams).to(device)
    module = net
    
    # Create snn loss instance.
    error = snn.loss(netParams, snn.loihi).to(device)
    
    # Define optimizer module.
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)
    
    # Create training and testing dataset
    trainingSet = snn.auto.dataset(
        dataset = nmnistDataset('NMNISTsmall', train=True),
        network = net,
    )
    testingSet = snn.auto.dataset(
        dataset = nmnistDataset('NMNISTsmall', train=False),
        network = net,
    )
    
    # Create dataloaders
    trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=True, num_workers=4)
    testLoader  = DataLoader(dataset=testingSet,  batch_size=12, shuffle=True, num_workers=4)

    # Learning stats instance.
    stats = snn.utils.stats()

    # Visualize the input spikes (first five samples).
    for i in range(5):
        input, target, label = trainingSet[i]
        snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

    # Create training assistant
    assist = snn.auto.assistant(
        net=net, 
        trainLoader=trainLoader, 
        testLoader=testLoader, 
        error=lambda o, t, l: error.numSpikes(o, t), 
        optimizer=optimizer, 
        stats=stats, 
        showTimeSteps=True
    )

    # training loop
    for epoch in range(100):
        # train
        assist.train(epoch)
        
        # test
        assist.test(epoch)

        # Update stats.
        stats.update()

        # plot lerarning stats, save stats and model
        stats.plot(saveFig=True, path=trainedFolder + '/')
        module.gradFlow(path=trainedFolder + '/')
        if stats.testing.bestAccuracy is True:  torch.save(module.state_dict(), trainedFolder + '/' + modelName + '.pt')            
        stats.save(trainedFolder + '/')

    # load the best model
    module.load_state_dict(torch.load(trainedFolder + '/' + modelName + '.pt'))

    # export the model
    module.genModel(trainedFolder + '/' + modelName + '.net')
