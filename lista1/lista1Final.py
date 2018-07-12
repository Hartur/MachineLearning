from sklearn import datasets, neighbors, metrics, linear_model
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import pandas
import numpy
import time

from Utils import Utils
from Classifiers import Classifiers
from Bagging import Bagging
from RandomSubspace import RandomSubspace

utils = Utils()
classifiers = Classifiers()
bagging = Bagging()
randomSubspace = RandomSubspace()

#Constantes
SPLIT_RATE = 0.6
LISTA1 = open('ResultadosLista1.txt', 'a')
REPETITIONS = 10
CLASSIFIERS = ['SGD', 'KNN', 'BAYES']

#Read Files
def readFileDataSet(dataset_file):
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

#Main
def valuateClassifier(X, y, classifier):
	labels = numpy.unique(y)

	X_train, X_test, y_train, y_test = utils.split(X, y, SPLIT_RATE)

	if classifier == 'KNN':
		c = classifiers.getKNNClassifier(X_train, y_train, KNN_WEIGHT)
	elif classifier == 'SGD':
		c = classifiers.getSGDClassifier(X_train, y_train)
	elif classifier == 'BAYES':
		c = classifiers.getBayesClassifier(X_train, y_train)


	expected = y_test
	predicted = c.predict(X_test)

	precisionArray = utils.getPrecisionArray(labels, expected, predicted, True)
	return precisionArray

def runSingleClassifierTest(X, y, labels, classifier):
	print(classifier)
	numOfLabels = len(labels)
	precisionArray = numpy.zeros(numOfLabels)
	for i in range(0, REPETITIONS):
		nowPrecisionArray = valuateClassifier(X, y, classifier)
		precisionArray = [x + y for x, y in zip(precisionArray, nowPrecisionArray)]

	precisionArray = [x / REPETITIONS for x in precisionArray]
	LISTA1.write('\n\n' + classifier + '\nLabel;Precision\n')
	for i in range(0, len(labels)):
		LISTA1.write('(' + str(labels[i]) + ',' + format(precisionArray[i], '.3f') + ')')
		
	LISTA1.write('\n')
	
	av = sum(precisionArray)/len(precisionArray)
	LISTA1.write('average:{' + format(av, '.3f') + '}\n')
	print(format(av, '.3f'))

def runBaggingTest(X, y, labels, classifier):
	K = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	numOfLabels = len(labels)
	LISTA1.write('Bagging ' + classifier + '\n')
	start_time = time.clock()
	print('Bagging ' + classifier)
	for k in K:
		precisionArray = numpy.zeros(numOfLabels)
		for i in range(0, REPETITIONS):
			nowPrecisionArray = bagging.valuateBaggingClassifier(X, y, k, classifier)
			precisionArray = [x + y for x, y in zip(precisionArray, nowPrecisionArray)]

		precisionArray = [x / REPETITIONS for x in precisionArray]
		av = sum(precisionArray)/len(precisionArray)
		print(format(av, '.3f'))
		LISTA1.write('(' + str(k) + ',' + format(av, '.3f') + ')')
	timeCounter = int(time.clock() - start_time)
	LISTA1.write("\n" + str(timeCounter) + " seconds\n")
	print(timeCounter, "seconds")

def runRandomSubspaceTest(X, y, labels, classifier, functionalExpansion):
	K = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	P = [30, 40, 50]
	numOfLabels = len(labels)
	LISTA1.write('Random Subspace ' + classifier + '\n')

	print("----------" + functionalExpansion + "----------\n")
	LISTA1.write("Functional Expansion " + functionalExpansion + "\n")
	if(functionalExpansion == 'powerSeries'):
		X = randomSubspace.FE_PowerSeries(X)
	elif(functionalExpansion == 'trigonometric'):
		X = randomSubspace.FE_Trigonometric(X)

	start_time = time.clock()
	print('Random Subspace ' + classifier)
	for p in P:
		LISTA1.write('\n' + str(p) + '% das features\n')
		for k in K:
			print('p: ' + str(p) + ' k: ' + str(k))
			precisionArray = numpy.zeros(numOfLabels)
			for i in range(0, REPETITIONS):
				nowPrecisionArray = randomSubspace.valuateRandomSubspace(X, y, k, p, classifier)
				precisionArray = [x + y for x, y in zip(precisionArray, nowPrecisionArray)]

			precisionArray = [x / REPETITIONS for x in precisionArray]
			av = sum(precisionArray)/len(precisionArray)

			print(format(av, '.3f'))
			LISTA1.write('(' + str(k) + ',' + format(av, '.3f') + ')')
	timeCounter = int(time.clock() - start_time) 
	LISTA1.write("\n" + str(timeCounter) + " seconds\n")
	print(timeCounter, "seconds")

def runClassifiers(X, y, dbName, functionalExpansion):
	print("----------" + dbName + "----------")
	LISTA1.write("----------" + dbName + "----------\n")
	labels = numpy.unique(y)
	numOfLabels = len(labels)
	
	for classifier in CLASSIFIERS:
		runSingleClassifierTest(X, y, labels, classifier)
		runBaggingTest(X, y, labels, classifier)
		runRandomSubspaceTest(X, y, labels, classifier, functionalExpansion)
		LISTA1.write('\n')

	LISTA1.write('\n\n')

def main():
	X, y = readFileDataSet('Skin_NonSkin.csv')
	runClassifiers(X, y, 'Skin Segmentation', 'trigonometric')

	X, y = readFileDataSet('BreastCancerDataSet.csv')
	runClassifiers(X, y, 'Breast Cancer', 'powerSeries')

	LISTA1.close()

main()


