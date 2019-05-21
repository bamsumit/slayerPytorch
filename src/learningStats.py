class learningStat():

	def __init__(self):
		self.lossSum = 0
		self.correctSamples = 0
		self.numSamples = 0
		self.minloss = None
		self.maxAccuracy = None

	def reset(self):
		self.lossSum = 0
		self.correctSamples = 0
		self.numSamples = 0

	def loss(self):
		if self.numSamples > 0:	
			return self.lossSum/self.numSamples	
		else:	
			return None

	def accuracy(self):
		if self.numSamples > 0 and self.correctSamples > 0:	
			return self.correctSamples/self.numSamples	
		else:	
			return None

	def update(self):
		currentloss = self.loss()
		if self.minloss is None:
			self.minloss = currentloss
		else:
			self.minloss = self.minloss if self.minloss < currentloss else currentloss

		currentAccuracy = self.accuracy()
		if self.maxAccuracy is None:
			self.maxAccuracy = currentAccuracy
		else:
			self.maxAccuracy = self.maxAccuracy if self.maxAccuracy > currentAccuracy else currentAccuracy

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
	def __init__(self):
		self.linesPrinted = 0
		self.training = learningStat()
		self.testing  = learningStat()

	def update(self):
		self.training.update()
		self.testing.update()

	def print(self, epoch, iter=None, timeElapsed=None):
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

	