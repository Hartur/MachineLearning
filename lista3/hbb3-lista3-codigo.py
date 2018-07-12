import numpy
from Utils import Utils
from Bagging import Bagging
from diversityMeasure import DiversityMeasure
from sklearn import metrics
import time

utils = Utils()
bagging = Bagging()

LISTA3 = open('ResultadosLista3.txt', 'w')
SPLIT_RATE = 0.6
REPETITIONS = 10



def writeResults(title, results):
	LISTA3.write('\n' + title + '\n')
	k=10
	for result in results:
		LISTA3.write('(' + str(k) + ', ' + format(result, '.3f') + ')')
		k = k + 10

def writeIndividualContribution(ic, k):
	LISTA3.write('\n\n Individual Contribution for ' + str(k) + ' classifiers\n')
	for i in ic:
		LISTA3.write('(' + str(i[0]) + ', ' + format(i[1], '.3f') + ')')


def evaluateEPIC_OLD(dbName):
	K = [20, 40, 60, 80, 100]
	S = [0.2, 0.4, 0.6, 0.8]

	X, y = utils.readFileDataSet(dbName+'.csv')

	for kIndex in range(0, len(K)):
		k = K[kIndex]
		print('------------' + str(k) + '------------')
		LISTA3.write('\n\n------------' + str(k) + '------------')
		precisionBagging = numpy.zeros(len(S))
		timmingBagging = numpy.zeros(len(S))
		precisionEPIC = numpy.zeros(len(S))
		timmingEPIC = numpy.zeros(len(S))

		for sIndex in range(0, len(S)):
			s = S[sIndex]
			numOfClassifiers = (int)(k * s)
			if(numOfClassifiers <= 1):
				continue
			print('--------' + str(s) + '--------')
			allTimeStart = time.clock()

			for rep in range(0, REPETITIONS):
				X_train, X_test, y_train, y_test = utils.split(X, y, SPLIT_RATE)

				start_time = time.clock()
				pool = bagging.getBaggingClassifier(X_train, y_train, k, 'TREE')
				predicted = bagging.predictBagging(X_test, pool, y_test)
				timmingBagging[sIndex] += int(time.clock() - start_time)
				prec = metrics.precision_score(y_test, predicted, average=None)
				p = sum(prec)/len(prec)
				precisionBagging[sIndex] += p

				diversityMeasure = DiversityMeasure(pool, X_test, y_test)

				individualContributions = []
				for i in range(0, k):
					individualContributions.append(diversityMeasure.getIndividualContribution(i))

				individualContributions = sorted(individualContributions, reverse= True, key= lambda ic: ic[1])

				selected = individualContributions[0:numOfClassifiers]
				newPool = []
				for classifier in selected:
					newPool.append(classifier[0])

				predicted = bagging.predictBagging(X_test, newPool, y_test)
				timmingEPIC += int(time.clock() - start_time)
				prec = metrics.precision_score(y_test, predicted, average=None)
				p = sum(prec)/len(prec)
				precisionEPIC[sIndex] += p

			precisionBagging[sIndex] /= REPETITIONS
			timmingBagging[sIndex] /= REPETITIONS
			precisionEPIC[sIndex] /= REPETITIONS
			timmingEPIC[sIndex] /= REPETITIONS
			print('=======================')
			print('Bagging precision: ' + format(precisionBagging[sIndex], '.3f'))
			print('Bagging timming: ' + str(timmingBagging[sIndex]))
			print('----')
			print('EPIC precision: ' + format(precisionEPIC[sIndex], '.3f'))
			print('EPIC timming: ' + str(timmingEPIC[sIndex]))
			print('=======================')
			print(str((int)(time.clock() - allTimeStart)) + ' seconds')

		LISTA3.write('\nBagging Precision\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + format(precisionBagging[sIndex], '.3f') + ')')

		LISTA3.write('\nBagging Timming\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + str(timmingBagging[sIndex]) + ')')

		LISTA3.write('\n------')

		LISTA3.write('\nEPIC Precision\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + format(precisionEPIC[sIndex], '.3f') + ')')

		LISTA3.write('\nEPIC Timming\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + str(timmingEPIC[sIndex]) + ')')

def evaluateEPIC(dbName):
	K = [20, 40, 60, 80, 100]
	S = [0.2, 0.4, 0.6, 0.8]

	X, y = utils.readFileDataSet(dbName+'.csv')

	for kIndex in range(0, len(K)):
		k = K[kIndex]
		print('------------' + str(k) + '------------')
		LISTA3.write('\n\n------------' + str(k) + '------------')
		precisionBagging = numpy.zeros(len(S))
		timmingBagging = numpy.zeros(len(S))
		precisionEPIC = numpy.zeros(len(S))
		timmingEPIC = numpy.zeros(len(S))

		for sIndex in range(0, len(S)):
			s = S[sIndex]
			numOfClassifiers = (int)(k * s)
			if(numOfClassifiers <= 1):
				continue
			print('--------' + str(s) + '--------')
			allTimeStart = time.clock()

			for rep in range(0, REPETITIONS):
				X_train, X_test, y_train, y_test = utils.split(X, y, SPLIT_RATE)

				start_time = time.clock()
				pool = bagging.getBaggingClassifier(X_train, y_train, k, 'SGD')
				predicted = bagging.predictBagging(X_test, pool, y_test)
				timmingBagging[sIndex] += int(time.clock() - start_time)
				prec = metrics.precision_score(y_test, predicted, average=None)
				p = sum(prec)/len(prec)
				precisionBagging[sIndex] += p

				X_train, X_pruning, y_train, y_pruning = utils.split(X_train, y_train, SPLIT_RATE)

				labels = numpy.unique(y).tolist()

				start_time = time.clock()
				pool = bagging.getBaggingClassifier(X_train, y_train, k, 'SGD')

				labelCount = numpy.zeros((len(X_pruning), len(y)))
				classifierPrediction = numpy.zeros((len(pool), len(X_pruning)))

				for i in range(0, len(pool)):
					for j in range(0, len(X_pruning)):
						prediction = pool[i].predict([X_pruning[j]])
						classifierPrediction[i][j] = prediction
						labelIndex = labels.index(prediction)
						labelCount[j][labelIndex] += 1

				diversityMeasure = DiversityMeasure(pool, X_test, y_test)

				individualContributions = []
				for i in range(0, len(pool)):
					result = 0
					for j in range(0, len(X_pruning)):
						alpha = 0
						beta = 0
						theta = 0

						prediction = classifierPrediction[i][j]
						labelIndex = labels.index(prediction)
						xLabelCount = labelCount[j]

						if(prediction == y_pruning[j]):
							if(labelCount[j][labelIndex] > max(xLabelCount)):
								alpha = 1
							else:
								beta = 1
						else:
							theta = 1

						correctLabelIndex = labels.index(y_pruning[j])

						xLabelCount = labelCount[j]
						second = sorted(xLabelCount, reverse= True)[1]

						result += alpha * (2 * max(xLabelCount) - labelCount[j][labelIndex]) + beta * second + theta * (labelCount[j][correctLabelIndex] - labelCount[j][labelIndex] - max(xLabelCount))

					individualContributions.append((pool[i], result))

				individualContributions = sorted(individualContributions, reverse=True, key=lambda ic: ic[1])

				selected = individualContributions[0:numOfClassifiers]
				newPool = []
				for classifier in selected:
					newPool.append(classifier[0])

				predicted = bagging.predictBagging(X_test, newPool, y_test)
				timmingEPIC += int(time.clock() - start_time)
				prec = metrics.precision_score(y_test, predicted, average=None)
				p = sum(prec)/len(prec)
				precisionEPIC[sIndex] += p

			precisionBagging[sIndex] /= REPETITIONS
			timmingBagging[sIndex] /= REPETITIONS
			precisionEPIC[sIndex] /= REPETITIONS
			timmingEPIC[sIndex] /= REPETITIONS
			print('=======================')
			print('Bagging precision: ' + format(precisionBagging[sIndex], '.3f'))
			print('Bagging timming: ' + str(timmingBagging[sIndex]))
			print('----')
			print('EPIC precision: ' + format(precisionEPIC[sIndex], '.3f'))
			print('EPIC timming: ' + str(timmingEPIC[sIndex]))
			print('=======================')
			print(str((int)(time.clock() - allTimeStart)) + ' seconds')

		LISTA3.write('\nBagging Precision\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + format(precisionBagging[sIndex], '.3f') + ')')

		LISTA3.write('\nBagging Timming\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + str(timmingBagging[sIndex]) + ')')

		LISTA3.write('\n------')

		LISTA3.write('\nEPIC Precision\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + format(precisionEPIC[sIndex], '.3f') + ')')

		LISTA3.write('\nEPIC Timming\n')
		for sIndex in range(0, len(S)):
			LISTA3.write('(' + str(S[sIndex]) + ',' + str(timmingEPIC[sIndex]) + ')')


def main():
	evaluateEPIC('ForestTypes')
	LISTA3.close()

main()