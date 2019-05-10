import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from spikeFileIO import *
from matplotlib import pyplot as plt

spikesFolder = 'test_files/binarySpikeTestFiles/'

# 1D num spike file
nID, tSt, tEn, nSp = read1DnumSpikes(spikesFolder + 'Out_0.bsn')
print(nID)
print(tSt)
print(tEn)
print(nSp)
encode1DnumSpikes(spikesFolder + 'Copy.bsn', nID, tSt, tEn, nSp)
nID_, tSt_, tEn_, nSp_ = read1DnumSpikes(spikesFolder + 'Copy.bsn')

print('1D num spike files read write:')
print('Error :', np.sum(np.abs(nID - nID_)))
print('Error :', np.sum(np.abs(tSt - tSt_)))
print('Error :', np.sum(np.abs(tEn - tEn_)))
print('Error :', np.sum(np.abs(nSp - nSp_)))

# 1D num spike file
TD = read1Dspikes(spikesFolder + 'NTIDIGITS_1.bs1')
print(TD.dim)
print(TD.x[:10])
print(TD.y)
print(TD.p[:10])
print(TD.t[:10])
encode1Dspikes(spikesFolder + 'Copy.bs1', TD)
TDcopy = read1Dspikes(spikesFolder + 'Copy.bs1')

print('1D spike files read write:')
print('Error :', np.sum(np.abs(TD.x - TDcopy.x)))
# print('Error :', np.sum(np.abs(TD.y - TDcopy.y)))
print('Error :', np.sum(np.abs(TD.p - TDcopy.p)))
print('Error :', np.sum(np.abs(TD.t - TDcopy.t)))

showTD(TD, 120)

# 2D binary spike file
TD = read2Dspikes(spikesFolder + 'NMNIST_1.bs2')
print(TD.dim)
print(TD.x[:10])
print(TD.y[:10])
print(TD.p[:10])
print(TD.t[:10])
encode2Dspikes(spikesFolder + 'Copy.bs2', TD)
TDcopy = read2Dspikes(spikesFolder + 'Copy.bs2')

print('2D spike files read write:')
print('Error :', np.sum(np.abs(TD.x - TDcopy.x)))
print('Error :', np.sum(np.abs(TD.y - TDcopy.y)))
print('Error :', np.sum(np.abs(TD.p - TDcopy.p)))
print('Error :', np.sum(np.abs(TD.t - TDcopy.t)))

# TD to spikeMat
spike = TD.toSpikeArray()
TDcopy = spikeArrayToEvent(spike)

plt.figure(1)
plt.imshow(spike.reshape((-1, spike.shape[-1])))

plt.figure(2)
# plt.plot(TD.t, 
# 		 TD.x + spike.shape[1] * (
# 		 TD.y + spike.shape[2] * 
# 		 TD.p), 'o')
plt.plot(np.round(TD.t), 
		 TD.x + spike.shape[1] * (
		 TD.y + spike.shape[2] * 
		 TD.p), 'o')
plt.plot(TDcopy.t, 
	     TDcopy.x + spike.shape[1] * (
	     TDcopy.y + spike.shape[2] * 
	     TDcopy.p), '.')
plt.show()


TD.t += 10000	# this should not effect showTD
showTD(TD, 120)