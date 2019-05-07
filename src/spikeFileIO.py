import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm

class event():
	def __init__(self, xEvent, yEvent, pEvent, tEvent):
		if yEvent is None:
			self.dim = 1
		else:
			self.dim = 2

		self.x = xEvent if type(xEvent) is np.array else np.asarray(xEvent) # x spatial dimension
		self.y = yEvent if type(yEvent) is np.array else np.asarray(yEvent) # y spatial dimension
		self.p = pEvent if type(pEvent) is np.array else np.asarray(pEvent) # spike polarity
		self.t = tEvent if type(tEvent) is np.array else np.asarray(tEvent) # time stamp in ms

	def toSpikeMat(self, samplingTime=1, dim=None):	# Sampling time in ms
		if self.dim == 1:
			if dim is None: dim = ( np.round(max(self.p)+1).astype(int),
									np.round(max(self.x)+1).astype(int), 
									np.round(max(self.t)/samplingTime+1).astype(int) )
			frame = np.zeros((dim[0], 1, dim[1], dim[2]))
		elif self.dim == 2:
			if dim is None: dim = ( np.round(max(self.p)+1).astype(int), 
									np.round(max(self.y)+1).astype(int), 
									np.round(max(self.x)+1).astype(int), 
									np.round(max(self.t)/samplingTime+1).astype(int) )
			frame = np.zeros((dim[0], dim[1], dim[2], dim[3]))
		return self.toSpikeTensor(frame, samplingTime).reshape(dim)

	def toSpikeTensor(self, emptyTensor, samplingTime=1):	# Sampling time in ms
		xEvent = np.round(self.x).astype(int)
		yEvent = np.round(self.y).astype(int)
		pEvent = np.round(self.p).astype(int)
		tEvent = np.round(self.t/samplingTime).astype(int)
		if self.dim == 1:
			validInd = np.argwhere((xEvent < emptyTensor.shape[2]) &
								   (pEvent < emptyTensor.shape[0]) &
								   (tEvent < emptyTensor.shape[3]))
			emptyTensor[pEvent[validInd],
						0, 
				  		xEvent[validInd],
				  		tEvent[validInd]] = 1/samplingTime
		elif self.dim == 2:
			validInd = np.argwhere((xEvent < emptyTensor.shape[2]) &
								   (yEvent < emptyTensor.shape[1]) & 
								   (pEvent < emptyTensor.shape[0]) &
								   (tEvent < emptyTensor.shape[3]))
			emptyTensor[pEvent[validInd], 
				  		yEvent[validInd],
				  		xEvent[validInd],
				  		tEvent[validInd]] = 1/samplingTime
		return emptyTensor

def numpyToEvent(spikeMat, samplingTime=1):
	if spikeMat.ndim == 3:
		spikeEvent = np.argwhere(spikeMat > 0)
		xEvent = spikeEvent[:,1]
		yEvent = None
		pEvent = spikeEvent[:,0]
		tEvent = spikeEvent[:,2]
	elif spikeMat.ndim == 4:
		spikeEvent = np.argwhere(spikeMat > 0)
		xEvent = spikeEvent[:,2]
		yEvent = spikeEvent[:,1]
		pEvent = spikeEvent[:,0]
		tEvent = spikeEvent[:,3]
	else:
		raise Exception('Expected numpy array of 3 or 4 dimension. It was {}'.format(spikeMat.ndim))

	return event(xEvent, yEvent, pEvent, tEvent * samplingTime) 

def read1Dspikes(filename):
	with open(filename, 'rb') as inputFile:
		inputByteArray = inputFile.read()
	inputAsInt = np.asarray([x for x in inputByteArray])
	xEvent =  (inputAsInt[0::5] << 8)  |  inputAsInt[1::5]
	pEvent =   inputAsInt[2::5] >> 7
	tEvent =( (inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5]) ) & 0x7FFFFF
	return event(xEvent, None, pEvent, tEvent/1000)	# convert spike times to ms

def encode1Dspikes(filename, TD):
	if TD.dim != 1: 	raise Exception('Expected Td dimension to be 1. It was: {}'.format(TD.dim))
	xEvent = np.round(TD.x).astype(int)
	pEvent = np.round(TD.p).astype(int)
	tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
	outputByteArray = bytearray(len(tEvent) * 5)
	outputByteArray[0::5] = np.uint8( (xEvent >> 8) & 0xFF00 ).tobytes()
	outputByteArray[1::5] = np.uint8( (xEvent & 0xFF) ).tobytes()
	outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
	outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
	outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()
	with open(filename, 'wb') as outputFile:
		outputFile.write(outputByteArray)

def read2Dspikes(filename):
	with open(filename, 'rb') as inputFile:
		inputByteArray = inputFile.read()
	inputAsInt = np.asarray([x for x in inputByteArray])
	xEvent =   inputAsInt[0::5]
	yEvent =   inputAsInt[1::5]
	pEvent =   inputAsInt[2::5] >> 7
	tEvent =( (inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5]) ) & 0x7FFFFF
	return event(xEvent, yEvent, pEvent, tEvent/1000)	# convert spike times to ms

def encode2Dspikes(filename, TD):
	if TD.dim != 2: 	raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
	xEvent = np.round(TD.x).astype(int)
	yEvent = np.round(TD.y).astype(int)
	pEvent = np.round(TD.p).astype(int)
	tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
	outputByteArray = bytearray(len(tEvent) * 5)
	outputByteArray[0::5] = np.uint8(xEvent).tobytes()
	outputByteArray[1::5] = np.uint8(yEvent).tobytes()
	outputByteArray[2::5] = np.uint8(((tEvent >> 16) & 0x7F) | (pEvent.astype(int) << 7) ).tobytes()
	outputByteArray[3::5] = np.uint8( (tEvent >> 8 ) & 0xFF ).tobytes()
	outputByteArray[4::5] = np.uint8(  tEvent & 0xFF ).tobytes()
	with open(filename, 'wb') as outputFile:
		outputFile.write(outputByteArray)

def read3Dspikes(filename):
	with open(filename, 'rb') as inputFile:
		inputByteArray = inputFile.read()
	inputAsInt = np.asarray([x for x in inputByteArray])
	xEvent =  (inputAsInt[0::7] << 4 ) | (inputAsInt[1::7] >> 4 )
	yEvent =  (inputAsInt[2::7] )    | ( (inputAsInt[1::7] & 0x0F) << 8 )
	pEvent =   inputAsInt[3::7]
	tEvent =( (inputAsInt[4::7] << 16) | (inputAsInt[5::7] << 8) | (inputAsInt[6::7]) )
	return event(xEvent, yEvent, pEvent, tEvent/1000)	# convert spike times to ms

def encode3Dspikes(filename, TD):
	if TD.dim != 2: 	raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
	xEvent = np.round(TD.x).astype(int)
	yEvent = np.round(TD.y).astype(int)
	pEvent = np.round(TD.p).astype(int)
	tEvent = np.round(TD.t * 1000).astype(int)	# encode spike time in us
	outputByteArray = bytearray(len(tEvent) * 7)
	outputByteArray[0::7] = np.uint8(xEvent >> 4).tobytes()
	outputByteArray[1::7] = np.uint8( ((xEvent << 4) & 0xFF) | (yEvent >> 8) & 0xFF00 ).tobytes()
	outputByteArray[2::7] = np.uint8(	yEvent & 0xFF ).tobytes()
	outputByteArray[3::7] = np.uint8(   pEvent ).tobytes()
	outputByteArray[4::7] = np.uint8(  (tEvent >> 16 ) & 0xFF ).tobytes()
	outputByteArray[5::7] = np.uint8(  (tEvent >> 8 ) & 0xFF ).tobytes()
	outputByteArray[6::7] = np.uint8(   tEvent & 0xFF ).tobytes()
	with open(filename, 'wb') as outputFile:
		outputFile.write(outputByteArray)

def read1DnumSpikes(filename):
	with open(filename, 'rb') as inputFile:
		inputByteArray = inputFile.read()
	inputAsInt = np.asarray([x for x in inputByteArray])
	neuronID =  (inputAsInt[0::10] << 8)  |  inputAsInt[1::10]
	tStart   =  (inputAsInt[2::10] << 16) | (inputAsInt[3::10] << 8) | (inputAsInt[4::10])
	tEnd     =  (inputAsInt[5::10] << 16) | (inputAsInt[6::10] << 8) | (inputAsInt[7::10])
	nSpikes  =  (inputAsInt[8::10] << 8)  |  inputAsInt[9::10]
	return neuronID, tStart/1000, tEnd/1000, nSpikes	# convert spike times to ms

def encode1DnumSpikes(filename, nID, tSt, tEn, nSp):
	neuronID = np.round(nID).astype(int)
	tStart   = np.round(tSt * 1000).astype(int)	# encode spike time in us
	tEnd     = np.round(tEn * 1000).astype(int)	# encode spike time in us
	nSpikes  = np.round(nSp).astype(int)
	outputByteArray = bytearray(len(neuronID) * 10)
	outputByteArray[0::10] = np.uint8( neuronID >> 8  ).tobytes()
	outputByteArray[1::10] = np.uint8( neuronID       ).tobytes()
	outputByteArray[2::10] = np.uint8( tStart   >> 16 ).tobytes()
	outputByteArray[3::10] = np.uint8( tStart   >> 8  ).tobytes()
	outputByteArray[4::10] = np.uint8( tStart         ).tobytes()
	outputByteArray[5::10] = np.uint8( tEnd     >> 16 ).tobytes()
	outputByteArray[6::10] = np.uint8( tEnd     >> 8  ).tobytes()
	outputByteArray[7::10] = np.uint8( tEnd           ).tobytes()
	outputByteArray[8::10] = np.uint8( nSpikes  >> 8  ).tobytes()
	outputByteArray[9::10] = np.uint8( nSpikes        ).tobytes()
	with open(filename, 'wb') as outputFile:
		outputFile.write(outputByteArray)

def _showTD1D(TD, frameRate=24, preComputeFrames=True, repeat=False):
	if TD.dim !=1:	raise Exception('Expected Td dimension to be 1. It was: {}'.format(TD.dim))
	fig = plt.figure()
	interval = 1e3 / frameRate					# in ms
	xDim = TD.x.max()+1
	tMax = TD.t.max()
	tMin = TD.t.min()
	pMax = TD.p.max()+1
	minFrame = int(np.floor(tMin / interval))
	maxFrame = int(np.ceil(tMax / interval )) + 1

	# ignore preComputeFrames

	def animate(i):
		fig.clear()
		tEnd = (i + minFrame + 1) * interval
		ind  = (TD.t < tEnd)
		# plot raster
		plt.plot(TD.t[ind], TD.x[ind], '.')
		# plt.plot(TD.t[ind], TD.x[ind], '.', c=cm.hot(TD.p[ind]))
		# plot raster scan line
		plt.plot([tEnd + interval, tEnd + interval], [0, xDim])
		plt.axis((tMin -0.1*tMax, 1.1*tMax, -0.1*xDim, 1.1*xDim))
		plt.draw()


	anim = animation.FuncAnimation(fig, animate, frames=maxFrame, interval=42, repeat=repeat) # 42 means playback at 23.809 fps

	plt.show()

def _showTD2D(TD, frameRate=24, preComputeFrames=True, repeat=False):
	if TD.dim != 2: 	raise Exception('Expected Td dimension to be 2. It was: {}'.format(TD.dim))
	fig = plt.figure()
	interval = 1e3 / frameRate					# in ms
	xDim = TD.x.max()+1
	yDim = TD.y.max()+1
	
	if preComputeFrames is True:
		minFrame = int(np.floor(TD.t.min() / interval))
		maxFrame = int(np.ceil(TD.t.max() / interval ))
		image    = plt.imshow(np.zeros((yDim, xDim, 3)))
		frames   = np.zeros( (maxFrame-minFrame, yDim, xDim, 3))

		# precompute frames
		for i in range(len(frames)):
			tStart = (i + minFrame) * interval
			tEnd = (i + minFrame + 1) * interval
			timeMask = (TD.t >= tStart) & (TD.t < tEnd)
			rInd = (timeMask & (TD.p == 1))
			gInd = (timeMask & (TD.p == 2))
			bInd = (timeMask & (TD.p == 0))
			frames[i, TD.y[rInd], TD.x[rInd], 0] = 1
			frames[i, TD.y[gInd], TD.x[gInd], 1] = 1
			frames[i, TD.y[bInd], TD.x[bInd], 2] = 1

		def animate(frame):
			image.set_data(frame)
			return image

		anim = animation.FuncAnimation(fig, animate, frames=frames, interval=42, repeat=repeat)

	else:
		minFrame = int(np.floor(TD.t.min() / interval))
		def animate(i):
			tStart = (i + minFrame) * interval
			tEnd = (i + minFrame + 1) * interval
			frame  = np.zeros((yDim, xDim, 3))
			timeMask = (TD.t >= tStart) & (TD.t < tEnd)
			rInd = (timeMask & (TD.p == 1))
			gInd = (timeMask & (TD.p == 2))
			bInd = (timeMask & (TD.p == 0))
			frame[TD.y[rInd], TD.x[rInd], 0] = 1
			frame[TD.y[gInd], TD.x[gInd], 1] = 1
			frame[TD.y[bInd], TD.x[bInd], 2] = 1
			plot = plt.imshow(frame)
			return plot

		anim = animation.FuncAnimation(fig, animate, interval=42, repeat=repeat) # 42 means playback at 23.809 fps

	# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
	# # installed.  The extra_args ensure that the x264 codec is used, so that
	# # the video can be embedded in html5.  You may need to adjust this for
	# # your system: for more information, see
	# # http://matplotlib.sourceforge.net/api/animation_api.html
	# if saveAnimation: anim.save('showTD_animation.mp4', fps=30)

	plt.show()

def showTD(TD, frameRate=24, preComputeFrames=True, repeat=False):
	if TD.dim == 1:
		_showTD1D(TD, frameRate=frameRate, preComputeFrames=preComputeFrames, repeat=repeat)		
	else:
		_showTD2D(TD, frameRate=frameRate, preComputeFrames=preComputeFrames, repeat=repeat)


def spikeMat2TD(spikeMat, samplingTime=1):		# Sampling time in ms
	addressEvent = np.argwhere(spikeMat > 0)
	# print(addressEvent.shape)
	return event(addressEvent[:,2], addressEvent[:,1], addressEvent[:,0], addressEvent[:,3] * samplingTime)
