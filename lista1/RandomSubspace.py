import numpy
import random
from Utils import Utils
from Classifiers import Classifiers


class RandomSubspace:
	
	KNN_WEIGHT = 'distance'
	SPLIT_RATE = 0.6
	utils = None
	classifiers = None

	def __init__(self):
		self.utils = Utils()
		self.classifiers = Classifiers()

	def getFeatures(self, X, featuresList):
		features = []
		for feature in featuresList:
			features.append(X[feature])
		return features

	def runRandomSubspace(self, X, pool):
		y = numpy.zeros(len(pool))

		for i in range(0, len(pool)):
			x = getFeatures(X, pool[i][0])
			y[i] = pool[i][1].predict([x])

		maxRepeticoes = 0
		moda = 0
		yToList = y.tolist()

		for i in yToList:
			repeticoes = yToList.count(i)
			if repeticoes > maxRepeticoes:
				maxRepeticoes = repeticoes
				moda = i

		return moda

	def predictRandomSubspace(self, X, pool, expected):
		predicted = []

		for x in X:
			predicted.append(self.runRandomSubspace(x, pool))

		return predicted

	def getRandomSubspace(self, X, y, k, p, classifier):
		pool = []
		numOfSamples = len(y)
		numOfFeatures = len(X[0])
		#Quantidade de atributos que serão usados
		numOfTrainFeatures = int(p * numOfFeatures / 100)

		featuresLists = []
		#Para todos os tipos de classificadores que devem ser criados
		for i in range(0, k):
			#Reinicia o array de features
			features = self.utils.generateArray(numOfFeatures-1)
			#Adicionar uma lista de features
			featuresLists.append([])
			#Coletar features
			for f in range(0, numOfTrainFeatures):
				pos = random.randint(0, len(features)-1)
				featuresLists[len(featuresLists) - 1].append(features[pos])
				features.pop(pos)

		#Para cada lista de features
		for fl in featuresLists:
			#Instanciar matriz de features
			X_train = []
			#Para cada exemplo da base de dados
			for x in X:
				#Preencher nova entrada com as posições das features sorteadas anteriormente
				X_train.append(getFeatures(x, fl))

			if classifier == 'KNN':
				c = self.classifiers.getKNNClassifier(X_train, y, self.KNN_WEIGHT)
			elif classifier == 'SGD':
				c = self.classifiers.getSGDClassifier(X_train, y)
			elif classifier == 'BAYES':
				c = self.classifiers.getBayesClassifier(X_train, y)
			
			pool.append((fl, c))

		return pool

	def valuateRandomSubspace(self, X, y, k, p, classifier):
		labels = numpy.unique(y)
		X_train, X_test, y_train, y_test = self.utils.split(X, y, self.SPLIT_RATE)
		expected = y_test
		pool = self.getRandomSubspace(X_train, y_train, k, p, classifier)

		predicted = self.predictRandomSubspace(X_test, pool, expected)
		precisionArray = self.utils.getPrecisionArray(labels, expected, predicted)
		return precisionArray

	#Functional Expansions
	def FE_PowerSeries(self, X):
		numOfSamples = len(X)
		numOfFeatures = len(X[0])
		for i in range(0, numOfSamples):
			for j in range(0,numOfFeatures):
				X[i].append(numpy.power(X[i][j], 2))
				X[i].append(numpy.power(X[i][j], 3))
		return X

	def FE_Trigonometric(self, X):
		numOfSamples = len(X)
		numOfFeatures = len(X[0])
		for i in range(0, numOfSamples):
			for j in range(0,numOfFeatures):
				X[i].append(numpy.cos(numpy.pi*X[i][j]/2))
				X[i].append(numpy.sin(numpy.pi*X[i][j]/2))
				X[i].append(numpy.cos(numpy.pi*X[i][j]))
				X[i].append(numpy.sin(numpy.pi*X[i][j]))
				X[i].append(numpy.cos(3*numpy.pi*X[i][j]/2))
				X[i].append(numpy.sin(3*numpy.pi*X[i][j]/2))
		return X