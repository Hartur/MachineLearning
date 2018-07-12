import numpy
import random
from Utils import Utils
from Classifiers import Classifiers

class Bagging:
	KNN_WEIGHT = 'distance'
	SPLIT_RATE = 0.6
	utils = None
	classifiers = None

	def __init__(self):
		self.utils = Utils()
		self.classifiers = Classifiers()

	def runBaggingClassifiers(self, X, pool):
		y = numpy.zeros(len(pool))

		for i in range(0, len(pool)):
			y[i] = pool[i].predict([X])

		maxRepeticoes = 0
		moda = 0
		yToList = y.tolist()

		for i in yToList:
			repeticoes = yToList.count(i)
			if repeticoes > maxRepeticoes:
				maxRepeticoes = repeticoes
				moda = i

		return moda

	def predictBagging(self, X, pool, expected):
		predicted = []

		for x in X:
			predicted.append(self.runBaggingClassifiers(x, pool))

		return predicted

	def getBaggingClassifier(self, X, y, k, classifier):
		pool = []
		numOfSamples = len(y)
		numOfFeatures = len(X[0])
		# Verifica quantos exemplos serão utilizados para gerar cada classificador (60% do total)
		numOfTrainSamples = int(60 * numOfSamples / 100)

		# Geração de classificadores de acordo com a quantidade selecionada
		for i in range(0, k):
			X_train = []
			y_train = []
			# Seleciona os exemplos que serão utilizados para treinar o classificador
			for j in range(0, numOfTrainSamples):
				pos = random.randint(0, numOfSamples - 1)
				X_train.append(X[pos])
				y_train.append(y[pos])
			# Gera novo classificador com os exemplos selecionados
			if classifier == 'KNN':
				c = self.classifiers.getKNNClassifier(X_train, y_train, self.KNN_WEIGHT)
			elif classifier == 'SGD':
				c = self.classifiers.getSGDClassifier(X_train, y_train)
			elif classifier == 'BAYES':
				c = self.classifiers.getBayesClassifier(X_train, y_train)
			elif classifier == 'TREE':
				c = self.classifiers.getDecisionTreeClassifier(X_train, y_train)
			# Adiciona o classificador ao pool de classificadores
			pool.append(c)

		return pool

	def valuateBaggingClassifier(self, X, y, k, classifier):
		labels = numpy.unique(y)
		X_train, X_test, y_train, y_test = self.utils.split(X, y, self.SPLIT_RATE)
		expected = y_test
		pool = self.getBaggingClassifier(X_train, y_train, k, classifier)
		predicted = self.predictBagging(X_test, pool, expected)

		precisionArray = self.utils.getPrecisionArray(labels, expected, predicted)
		return precisionArray
