from sklearn import metrics
import random
import pandas
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

	def readFileDataSet(self, dataset_file):
		dataSet = pandas.read_csv(dataset_file, header = None, sep = ';').to_records()
		numOfSamples = len(dataSet)
		#Quantidade de features calculada pela quantidade total de colunas menos
		#uma do número de ordem da linha (gerado pelo pandas durante a leitura), 
		#e uma do label (última coluna de informações da base)
		numOfFeatures = len(dataSet[0]) - 2

		X = []
		y = []
		yColumn = len(dataSet[0]) - 1
		#Para todos os exemplos
		for i in range(0, numOfSamples):
			X.append([])
			#Adicionar valor do label em y
			y.append(float(dataSet[i][yColumn]))
			#Para todas as colunas das features
			for j in range (1, numOfFeatures + 1):
				#Adicionar valor da feature em X
				X[len(X)-1].append(float(dataSet[i][j]))

		return X, y