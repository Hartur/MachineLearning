import numpy
import math
from DynamicSelectionUtils import DynamicSelectionUtils

class KNORAU:
	dynamicSelectionUtils = None

	def __init__(self):
		self.dynamicSelectionUtils = DynamicSelectionUtils()

	def predict(self, pool, nearest, X, y, Xquery):
		#Seleciona amostras mais próximas
		#nearest = self.dynamicSelectionUtils.getNearst(n, X, y, Xquery)
		classifiers = []
		#Calcula o peso dos classificadores que acertaram pelo menos 1 amostra
		for c in pool:
			weight = 0
			for n in nearest:
				if(c.predict([n[0]]) == n[1]):
					weight += 1

			if(weight > 0):
				classifiers.append((c, weight))

		labels = numpy.unique(y)
		result = []
		for l in labels:
			result.append((0, l))

		labels = labels.tolist()
		#Calcula a classe de acordo com os pesos dos classificadores
		for c in classifiers:
			index = labels.index(c[0].predict([Xquery]))
			weight = result[index][0] + c[1]
			result[index] = (weight, result[index][1])

		result = sorted(result, reverse=True, key=lambda r: r[0])
		return result[0][1]