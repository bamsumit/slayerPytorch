class learningStat():
	'''
	This class collect the learning statistics over the epoch.

	Usage:

	This class is designed to be used with learningStats instance although it can be used separately.

	>>> trainingStat = learningStat()
	'''
	def __init__(self):
		self.lossSum = 0
		self.correctSamples = 0
		self.numSamples = 0
		self.minloss = None
		self.maxAccuracy = None
		self.lossLog = []
		self.accuracyLog = []
		self.bestLoss = False
		self.bestAccuracy = False

	def reset(self):
		'''
		Reset the learning staistics. 
		This should usually be done before the start of an epoch so that new statistics counts can be accumulated.

		Usage:

		>>> trainingStat.reset()
		'''
		self.lossSum = 0
		self.correctSamples = 0
		self.numSamples = 0

	def loss(self):
		'''
		Returns the average loss calculated from the point the stats was reset.

		Usage:

		>>> loss = trainingStat.loss()
		'''
		if self.numSamples > 0:	
			return self.lossSum/self.numSamples	
		else:	
			return None

	def accuracy(self):
		'''
		Returns the average accuracy calculated from the point the stats was reset.

		Usage:

		>>> loss = trainingStat.accuracy()
		'''
		if self.numSamples > 0 and self.correctSamples > 0:	
			return self.correctSamples/self.numSamples	
		else:	
			return None

	def update(self):
		'''
		Updates the stats of the current session and resets the measures for next session.

		Usage:

		>>> trainingStat.update()
		'''
		currentLoss = self.loss()
		self.lossLog.append(currentLoss)
		if self.minloss is None:
			self.minloss = currentLoss
		else:
			if currentLoss < self.minloss:
				self.minloss = currentLoss
				self.bestLoss = True
			else:
				self.bestLoss = False
			# self.minloss = self.minloss if self.minloss < currentLoss else currentLoss

		currentAccuracy = self.accuracy()
		self.accuracyLog.append(currentAccuracy)
		if self.maxAccuracy is None:
			self.maxAccuracy = currentAccuracy
		else:
			if currentAccuracy > self.maxAccuracy:
				self.maxAccuracy = currentAccuracy
				self.bestAccuracy = True
			else:
				self.bestAccuracy = False
			# self.maxAccuracy = self.maxAccuracy if self.maxAccuracy > currentAccuracy else currentAccuracy

	def displayString(self):
		loss = self.loss()
		accuracy = self.accuracy()
		minloss = self.minloss
		maxAccuracy = self.maxAccuracy

		if loss is None:	# no stats available
			return None
		elif accuracy is None: 
			if minloss is None:	# accuracy and minloss stats is not available
				return 'loss = %-12.5g'%(loss)
			else:	# accuracy is not available but minloss is available
				return 'loss = %-12.5g (min = %-12s)'%(loss, minloss)
		else:
			if minloss is None and maxAccuracy is None: # minloss and maxAccuracy is available
				return 'loss = %-12.5g        %-12s   \taccuracy = %-10.5g        %-10s '%(loss, ' ', accuracy, ' ')
			else:	# all stats are available
				return 'loss = %-12.5g (min = %-12.5g)  \taccuracy = %-10.5g (max = %-10.5g)'%(loss, minloss, accuracy, maxAccuracy)

class learningStats():
	'''
	This class provides mechanism to collect learning stats for training and testing, and displaying them efficiently.

	Usage:

	.. code-block:: python
	
		stats = learningStats()

		for epoch in range(100):
			tSt = datetime.now()

			stats.training.reset()
			for i in trainingLoop:
				# other main stuffs
				stats.training.correctSamples += numberOfCorrectClassification
				stats.training.numSamples     += numberOfSamplesProcessed
				stats.training.lossSum        += currentLoss
				stats.print(epoch, i, (datetime.now() - tSt).total_seconds())
			stats.training.update()

			stats.testing.reset()
			for i in testingLoop
				# other main stuffs
				stats.testing.correctSamples += numberOfCorrectClassification
				stats.testing.numSamples     += numberOfSamplesProcessed
				stats.testing.lossSum        += currentLoss
				stats.print(epoch, i)
			stats.training.update()

	'''
	def __init__(self):
		self.linesPrinted = 0
		self.training = learningStat()
		self.testing  = learningStat()

	def update(self):
		'''
		Updates the stats for training and testing and resets the measures for next session.

		Usage:

		>>> stats.update()
		'''
		self.training.update()
		self.training.reset()
		self.testing.update()
		self.testing.reset()

	def print(self, epoch, iter=None, timeElapsed=None):
		'''
		Prints the available learning statistics from the current session on the console.
		For Linux systems, prints the data on same terminal space (might not work properly on other systems).

		Arguments:
			* ``epoch``: epoch counter to display (required).
			* ``iter``: iteration counter to display (not required).
			* ``timeElapsed``: runtime information (not required).

		Usage:

		.. code-block:: python

			# prints stats with epoch index provided
			stats.print(epoch) 

			# prints stats with epoch index and iteration index provided
			stats.print(epoch, iter=i) 
			
			# prints stats with epoch index, iteration index and time elapsed information provided
			stats.print(epoch, iter=i, timeElapsed=time) 
		'''
		print('\033[%dA'%(self.linesPrinted))
		
		self.linesPrinted = 1

		epochStr   = 'Epoch : %10d'%(epoch)
		iterStr    = '' if iter is None else '(i = %7d)'%(iter)
		profileStr = '' if timeElapsed is None else ', %12.4f ms elapsed'%(timeElapsed * 1000)

		print(epochStr + iterStr + profileStr)
		print(self.training.displayString())
		self.linesPrinted += 2
		if self.testing.displayString() is not None:
			print(self.testing.displayString())
			self.linesPrinted += 1

	