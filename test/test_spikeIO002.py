import sys, os

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_TEST_DIR + "/../src")

import unittest
from spikeFileIO import *
import random

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

spikesFolder = 'test_files/binarySpikeTestFiles/'

i = random.randint(1, 10)

TD = read2Dspikes(spikesFolder + '{}.bs2'.format(i))
if verbose is True:
	if bool(os.environ.get('DISPLAY', None)):
		showTD(TD)

encode3Dspikes(spikesFolder + '{}.bs3'.format(i), TD)
TDcopy = read3Dspikes(spikesFolder + '{}.bs3'.format(i))
if verbose is True:
	if bool(os.environ.get('DISPLAY', None)):
		showTD(TDcopy)

class Test3DSpikesRW(unittest.TestCase):
	def test(self):
		if verbose is True:
			print('3D spike files read write:')
			print('Error x:', np.sum(np.abs(TD.x - TDcopy.x)))
			print('Error y:', np.sum(np.abs(TD.y - TDcopy.y)))
			print('Error p:', np.sum(np.abs(TD.p - TDcopy.p)))
			print('Error t:', np.sum(np.abs(TD.t - TDcopy.t)))
		self.assertEqual(np.sum(np.abs(TD.x - TDcopy.x)), 0, 'x Read and write values must match.')
		self.assertEqual(np.sum(np.abs(TD.y - TDcopy.y)), 0, 'y Read and write values must match.')
		self.assertEqual(np.sum(np.abs(TD.p - TDcopy.p)), 0, 'p Read and write values must match.')
		self.assertEqual(np.sum(np.abs(TD.t - TDcopy.t)), 0, 't Read and write values must match.')

if __name__ == '__main__':	
	unittest.main()