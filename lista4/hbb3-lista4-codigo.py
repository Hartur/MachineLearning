import numpy
from Utils import Utils
from DES import KNORAU
from DCS import OLA
from DCS import LCA
from DynamicSelectionUtils import DynamicSelectionUtils
from Bagging import Bagging
from sklearn import metrics
import time
import random

utils = Utils()
bagging = Bagging()
KNORAU = KNORAU()
OLA = OLA()
LCA = LCA()
dynamicSelectionUtils = DynamicSelectionUtils()

LISTA4 = open('ResultadosLista4.txt', 'w')
SPLIT_RATE = 0.6
REPETITIONS = 10
CLASSIFIER = 'SGD'

def writeResults(title, results):
	print(title)
	LISTA4.write('\n' + title + '\n')
	n=4
	for result in results:
		print('(' + str(n) + ', ' + format(result/REPETITIONS, '.3f') + ')')
		LISTA4.write('(' + str(n) + ', ' + format(result/REPETITIONS, '.3f') + ')')
		n = n + 2

def evaluateDynamicSelection(dbName):
	print('------------' + dbName + '------------')
	LISTA4.write('------------' + dbName + '------------\n')
	X, y = utils.readFileDataSet(dbName+'.csv')
	K = [20,40,60,80,100]
	nearest = [4,6,8]

	#Para cada quantidade de classificadores no pool gerado utilizando bagging
	for k in K:
		baggingPrecisionSum = 0
		knorauPrecisionSum = numpy.zeros(len(nearest))
		olaPrecisionSum = numpy.zeros(len(nearest))
		lcaPrecisionSum = numpy.zeros(len(nearest))
		print('------------' + str(k) + '------------')
		LISTA4.write('\n------------' + str(k) + ' Classificadores------------\n')

		for i in range(0, REPETITIONS):
			X_train, X_test, y_train, y_test = utils.split(X, y, SPLIT_RATE)
			X_train, X_DS, y_train, y_DS = utils.split(X_train, y_train, SPLIT_RATE)
			pool = bagging.getBaggingClassifier(X,y,k,CLASSIFIER)
			baggingPredict = []
			knorauPredict = [[]]
			olaPredict = [[]]
			lcaPredict = [[]]
			#Para cada quantidade de classificadores selecionados no KNORAU
			for n in range(0, len(nearest)-1):
				#Cria um array para guardar as previsões
				knorauPredict.append([])
				olaPredict.append([])
				lcaPredict.append([])

			#Para cada teste
			for i in range(0, len(X_test)):
				#Adiciona a previsão do bagging
				baggingPredict.append(bagging.predictBagging([X_test[i]], pool))
				#Para cada quantidade de classificadores selecionados no KNORAU
				for n in nearest:
					#Adiciona a previsão do KNORAU, OLA e LCA selecionando os n classificadores mais próximos e salva no seu respectivo array
					index = (int)(n/2)-2
					nearestSamples = dynamicSelectionUtils.getNearst(n, X, y, X_test[i])
					knorauPredict[index].append(KNORAU.predict(pool, nearestSamples, X_DS, y_DS, X_test[i]))
					olaPredict[index].append(OLA.predict(pool, nearestSamples, X_DS, y_DS, X_test[i]))
					lcaPredict[index].append(LCA.predict(pool, nearestSamples, X_DS, y_DS, X_test[i]))
			#Calcula a precisão do bagging
			baggingPrecision = metrics.precision_score(y_test, baggingPredict, average=None)
			baggingPrecision = sum(baggingPrecision)/len(baggingPrecision)
			#Soma resultado da precisão do bagging
			baggingPrecisionSum += baggingPrecision

			#Calcula a precisão do KNORAU, OLA e LCA para cada quantidade de classificadores selecionados
			for i in range(0, len(nearest)):
				prec = metrics.precision_score(y_test, knorauPredict[i], average=None)
				prec = sum(prec)/len(prec)
				knorauPrecisionSum[i] += prec

				prec = metrics.precision_score(y_test, olaPredict[i], average=None)
				prec = sum(prec)/len(prec)
				olaPrecisionSum[i] += prec

				prec = metrics.precision_score(y_test, lcaPredict[i], average=None)
				prec = sum(prec)/len(prec)
				lcaPrecisionSum[i] += prec
			
		print("Bagging Precision: " + format(baggingPrecisionSum/REPETITIONS, '.3f'))
		LISTA4.write("Bagging Precision: " + format(baggingPrecisionSum/REPETITIONS, '.3f'))
		writeResults("KNORAU Precision", knorauPrecisionSum)
		writeResults("OLA Precision", olaPrecisionSum)
		writeResults("LCA Precision", lcaPrecisionSum)



def main():
	evaluateDynamicSelection('ForestTypes')
	evaluateDynamicSelection('BreastCancerDataSet')
	evaluateDynamicSelection('Skin_NonSkin')
	LISTA4.close()

main()