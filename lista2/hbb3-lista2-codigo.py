import numpy
from Utils import Utils
from Bagging import Bagging
from diversityMeasure import DiversityMeasure
from sklearn import metrics

utils = Utils()
bagging = Bagging()

LISTA2 = open('ResultadosLista2.txt', 'w')
SPLIT_RATE = 0.6
REPETITIONS = 10

def writeResults(title, results):
	LISTA2.write('\n' + title + '\n')
	k=10
	for result in results:
		LISTA2.write('(' + str(k) + ', ' + format(result, '.3f') + ')')
		k = k + 10


def getMeasures(dbName):
	K = [10,20,30,40,50,60,70,80,90,100]
	precision = numpy.zeros(len(K))
	doubleFaultDiversity = numpy.zeros(len(K))
	disagreementDiversity = numpy.zeros(len(K))
	entropyDiversity = numpy.zeros(len(K))
	coincidentFailureDiversity = numpy.zeros(len(K))

	LISTA2.write('\n\n----------' + dbName + '----------\n')
	print('----------' + dbName + '----------\n')

	X, y = utils.readFileDataSet(dbName+'.csv')

	for i in range(0,REPETITIONS):
		print('----------------' + dbName + ' ' + str(i) + '----------------')
		X_train, X_test, y_train, y_test = utils.split(X, y, SPLIT_RATE)

		for j in range(0, len(K)):
			k = K[j]
			print('---k: ' + str(k) + '---')

			pool = bagging.getBaggingClassifier(X_train, y_train, k, 'SGD')
			predicted = bagging.predictBagging(X_test, pool, y_test)

			diversityMeasure = DiversityMeasure(pool, X_test, y_test)

			prec = metrics.precision_score(y_test, predicted, average=None)
			p = sum(prec)/len(prec)
			print('Precision: ' + format(p, '.3f'))
			precision[j] += p
			print('Precision sum: ' + format(precision[j], '.3f'))

			df = diversityMeasure.doubleFaultMeasure()
			print('Double Fault: ' + format(df,'.3f'))
			doubleFaultDiversity[j] += df
			print('Double Fault sum: ' + format(doubleFaultDiversity[j],'.3f'))

			d = diversityMeasure.disagreementMeasure()
			print('Disagreement: ' + format(d,'.3f'))
			disagreementDiversity[j] += d
			print('Disagreement sum: ' + format(disagreementDiversity[j],'.3f'))

			e = diversityMeasure.entropyMeasure()
			print('Entropy: ' + format(e,'.3f'))
			entropyDiversity[j] += e
			print('Entropy sum: ' + format(entropyDiversity[j],'.3f'))

			cfd = diversityMeasure.coincidentFailureDiversity()
			print('Coincident Failure Diversity: ' + format(d,'.3f'))
			coincidentFailureDiversity[j] += cfd
			print('Coincident Failure Diversity sum: ' + format(coincidentFailureDiversity[j],'.3f'))
	
	precision = [x/REPETITIONS for x in precision]
	doubleFaultDiversity = [x/REPETITIONS for x in doubleFaultDiversity]
	disagreementDiversity = [x/REPETITIONS for x in disagreementDiversity]
	entropyDiversity = [x/REPETITIONS for x in entropyDiversity]
	coincidentFailureDiversity = [x/REPETITIONS for x in coincidentFailureDiversity]

	relation = [x / y for x, y in zip(doubleFaultDiversity, disagreementDiversity)]

	LISTA2.write('\n----------MEASURES----------\n')
	writeResults('Precision', precision)
	writeResults('Double Fault', doubleFaultDiversity)
	writeResults('Disagreement', disagreementDiversity)
	writeResults('Entropy Measure', entropyDiversity)
	writeResults('Coincident Failure', coincidentFailureDiversity)
	writeResults('Relation', relation)


	LISTA2.write('\n----------MOVED MEASURES----------\n')
	p = max(precision)
	df = max(doubleFaultDiversity)
	d = max(disagreementDiversity)
	e = max(entropyDiversity)
	c = max(coincidentFailureDiversity)

	deslocamento = max(p, df, d, e, c)

	precision = [x + deslocamento - p for x in precision]
	doubleFaultDiversity = [x + deslocamento - df for x in doubleFaultDiversity]
	disagreementDiversity = [x + deslocamento - d for x in disagreementDiversity]
	entropyDiversity = [x + deslocamento - e for x in entropyDiversity]
	coincidentFailureDiversity = [x + deslocamento - c for x in coincidentFailureDiversity]

	writeResults('Precision Normalized', precision)
	writeResults('Double Fault Normalized', doubleFaultDiversity)
	writeResults('Disagreement Normalized', disagreementDiversity)
	writeResults('Entropy Measure Normalized', entropyDiversity)
	writeResults('Coincident Failure Normalized', coincidentFailureDiversity)


	LISTA2.write('\n----------RELATION BETWEEN MEASURES AND PRECISION----------\n')
	doubleFaultDiversity = [x / y for x, y in zip(doubleFaultDiversity, precision)]
	disagreementDiversity = [x / y for x, y in zip(disagreementDiversity, precision)]
	entropyDiversity = [x / y for x, y in zip(entropyDiversity, precision)]
	coincidentFailureDiversity = [x / y for x, y in zip(coincidentFailureDiversity, precision)]

	writeResults('Double Fault / Precision', doubleFaultDiversity)
	writeResults('Disagreement / Precision', disagreementDiversity)
	writeResults('Entropy Measure / Precision', entropyDiversity)
	writeResults('Coincident Failure / Precision', coincidentFailureDiversity)

def main():
	getMeasures('ForestTypes')
	LISTA2.close()

main()