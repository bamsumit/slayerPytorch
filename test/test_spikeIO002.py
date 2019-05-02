import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

from spikeFileIO import *
import random

spikesFolder = 'test_files/binarySpikeTestFiles/'

i = random.randint(1, 10)

TD = read2Dspikes(spikesFolder + '{}.bs2'.format(i))
showTD(TD)

encode3Dspikes(spikesFolder + '{}.bs3'.format(i), TD)
TDcopy = read3Dspikes(spikesFolder + '{}.bs3'.format(i))
showTD(TDcopy)

print('3D spike files read write:')
print('Error :', np.sum(np.abs(TD.x - TDcopy.x)))
print('Error :', np.sum(np.abs(TD.y - TDcopy.y)))
print('Error :', np.sum(np.abs(TD.p - TDcopy.p)))
print('Error :', np.sum(np.abs(TD.t - TDcopy.t)))