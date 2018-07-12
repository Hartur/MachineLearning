import numpy

class DiversityMeasure:
	pool = None
	X_test = None
	y_test = None

	def __init__(self, pool, X_test, y_test):
		self.pool = pool
		self.X_test = X_test
		self.y_test = y_test

	def vamove(self, c):
		if c:
			return 1
		else:
			return 0

	def getTableOfRelationship (self, alpha, beta):
		y_test = numpy.array(self.y_test)
		table = numpy.zeros((2,2));

		for i in range(0,len(y_test)):
			if(alpha[i] != y_test[i] and beta[i] != y_test[i]):
				table[0][0] = table[0][0] + 1
			elif(alpha[i] != y_test[i] and beta[i] == y_test[i]):
				table[0][1] = table[0][1] + 1
			elif(alpha[i] == y_test[i] and beta[i] != y_test[i]):
				table[1][0] = table[1][0] + 1
			elif(alpha[i] == y_test[i] and beta[i] == y_test[i]):
				table[1][1] = table[1][1] + 1

		return table

	def doubleFaultMeasure(self):
		numOfClassifiers = len(self.pool)
		correlationTable = numpy.zeros((numOfClassifiers, numOfClassifiers))
		dfm = []

		for i in range(0, numOfClassifiers):
			for j in range(0, i):
				table = self.getTableOfRelationship(self.pool[i].predict(self.X_test), self.pool[j].predict(self.X_test))
				dfm.append(table[0][0] / (table[0][0] + table[0][1] + table[1][0] + table[1][1]))

		return numpy.sum(dfm)/len(dfm)

	def disagreementMeasure(self):
		numOfClassifiers = len(self.pool)
		correlationTable = numpy.zeros((numOfClassifiers, numOfClassifiers))
		disagreement = []

		for i in range(0, numOfClassifiers):
			for j in range(0, i):
				table = self.getTableOfRelationship(self.pool[i].predict(self.X_test), self.pool[j].predict(self.X_test))
				disagreement.append((table[0][1] + table[1][0]) / (table[0][0] + table[0][1] + table[1][0] + table[1][1]))

		return numpy.sum(disagreement)/len(disagreement)

	def coincidentFailureDiversity(self):
		pi = numpy.zeros(len(self.pool) + 1)
		for i in range(0, len(self.y_test)):
			acertos = 0
			for c in self.pool:
				if c.predict([self.X_test[i]]) == self.y_test[i]:
					acertos += 1
			pi[acertos] += 1

		pi = [x/len(self.y_test) for x in pi]

		if pi[0] == 1:
			return 1
		else:
			L = len(self.pool)
			soma = 0
			for i in range(1,len(pi)):
				soma += ((L - i) / (L - 1)) * pi[i]
			return ((1 / (1 - pi[0])) * soma)

	def entropyMeasure(self):
		N = len(self.y_test)
		L = len(self.pool)

		soma = 0
		for i in range(0, N):
			acertos = 0
			erros = 0
			for c in self.pool:
				if c.predict([self.X_test[i]]) == self.y_test[i]:
					acertos += 1
				else:
					erros += 1
			soma += min(acertos, erros)

		return (2/(N*L)) * soma