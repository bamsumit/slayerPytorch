import slayer
import spikeio as sIO
from torch.autograd import Variable
from torch.utils.data import Dataset, Dataloader

torch.backends.cudnn.benchmark = True

# Dataloader definition
class nmnistDataset(Dataset):
	def __init__(self):
		data = np.loadtxt('..Path to file train.txt or test.txt..')
		self.len    = data.shape[0]
		self.input  = data[:, 0]
		self.output = data[:, 1]
		
	def __getitem__(self, index)
		spikeInput  = sIO.read('.. Path to spike file ...' + str(self.input[index]) )
		spikeOutput = sIO.read('.. Path to spike file ...' + str(self.output[index]) )
		return spikeInput, spikeOutput
		
	def __len__(self)
		return self.len
		
dataset = nmnistDataset()
trainLoader = DataLoader(dataset = dataset, batch_size = 16, shuffle = True, num_workers = 2)

# Network definition
class Network(torch.nn.Module):
	def __init__(self, netParam, device=torch.device('cuda')):
		# Constructor
		super(Network, self).__init__() # call the constructor of parent class
		
		self.netParam = netParam
		
		# self.spikeResponse = slayer.alphaKernel     (self.netParam['neuron']['tauSr'],  self.netParam['simulation']['Ts'])
		# self.refResponse   = slayer.refractoryKernel(self.netParam['neuron']['tauRef'], self.netParam['simulation']['Ts'])
		
		slayer = spikeLayer(net_params['neuron'], net_params['simulation'])
				
		self.spike  = slayer.spike()
		self.layer1 = slayer.dense((34, 34, 2), 512)
		self.layer2 = slayer.dense(512, 10)
		# self. layer1 = slayer.conv((5, 5, 8))
		# torch.nn.init.normal_(self.layer1.weight, mean=0, std=1)
		
	def forward(self, spikeInput):
		spikeLayer1 = self.spike(self.layer1(spikeInput))
		spikeLayer2 = self.spike(self.layer2(spikeLayer1))
		return spikeLayer2
		
# Network creation
net = Network()

# Cost function
costFunct = slayer.spikeDistance()

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)

# training loop
for epoch in range(500):
	for i, (input, target) in enumerate(trainloader, 0):
		output = net.forward(input)
		loss = costFunct(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		