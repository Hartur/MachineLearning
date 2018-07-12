from sklearn import metrics
import random
import numpy

class Utils:

	def mixSamples(self, X, y):
		size = len(y)
		for i in range(0, size*3):
			pos1 = random.randint(0, size-1)
			pos2 = random.randint(0, size-1)
			aux = X[pos1]
			X[pos1] = X[pos2]
			X[pos2] = aux

			aux = y[pos1]
			y[pos1] = y[pos2]
			y[pos2] = aux
		return X, y

	def split(self, X, y, portion):
		X, y = self.mixSamples(X, y)

		labels = numpy.unique(y)
		qtdTrain = []
		#Pega a quantidade de exemplos com determinada classe e calcula
		#a quantidade de testes que serão utilizados para treinar
		for l in labels:
			total = y.count(l)
			qtdTrain.append(int(total*60/100))

		X_train = []
		X_test = []
		y_train = []
		y_test = []
		#Para todos os exemplos existentes, separa os que serão utilizados 
		#para treinar e para testar
		labelsToList = labels.tolist()

		for i in range(0, len(y)):
			label_index = labelsToList.index(y[i])
			if(qtdTrain[label_index] != 0):
				qtdTrain[label_index] -= 1
				X_train.append(X[i])
				y_train.append(y[i])
			else:
				X_test.append(X[i])
				y_test.append(y[i])

		return X_train, X_test, y_train, y_test

	def getPrecisionArray(self, labels, expected, predicted, writeAll = False):
		precisionArray = metrics.precision_score(expected, predicted, average=None)
		return precisionArray

	def generateArray(self, qtd):
		a = []
		for i in range(0, qtd):
			a.append(i)
		return a
