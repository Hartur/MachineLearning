import numpy
from DynamicSelectionUtils import DynamicSelectionUtils
import random

class OLA:
	dynamicSelectionUtils = None
	bagging = None

	def __init__(self):
		self.dynamicSelectionUtils = DynamicSelectionUtils()

	def predict(self, pool, nearest, X, y, Xquery):
		#Seleciona amostras mais próximas
		#nearest = self.dynamicSelectionUtils.getNearst(n, X, y, Xquery)
		classifiers = []
		predicts = []
		#Calcula quantas amostras mais próximas cada classificador acertou
		for c in pool:
			weight = 0
			for n in nearest:
				p = c.predict([n[0]])
				predicts.append(p)
				if(p == n[1]):
					weight += 1

			if(weight > 0):
				classifiers.append((c, weight))

		#se todos concordarem, retorna o valor
		if(len(numpy.unique(predicts)) == 1):
			return predicts[0]

		#caso contrário, verifica qual classificador mais acerto
		result = sorted(classifiers, reverse=True, key=lambda c: c[1])
		classifier = result[0][0]

		return classifier.predict([Xquery])



class LCA:
	def __init__(self):
		self.dynamicSelectionUtils = DynamicSelectionUtils()

	def predict(self, pool, nearest, X, y, Xquery):
		#Seleciona amostras mais próximas
		#nearest = self.dynamicSelectionUtils.getNearst(n, X, y, Xquery)
		classifiers = []
		#Calcula quantas amostras mais próximas cada classificador acertou
		for c in pool:
			right = 0
			sameClass = 0
			yquery = c.predict([Xquery])

			for n in nearest:
				p = c.predict([n[0]])
				if(p == yquery):
					sameClass += 1
					if(p == n[1]):
						right += 1

			if(right > 0):
				classifiers.append((c, right/sameClass))

		#Verifica qual classificador teve maior taxa de acerto.
		if(len(classifiers) > 0):
			result = sorted(classifiers, reverse=True, key=lambda c: c[1])
			classifier = result[0][0]
			return classifier.predict([Xquery])
		else:
			#Caso nenhum tenha acertado, seleciona um classificador aleatóriamente
			index = random.randint(0, len(pool)-1)
			return pool[index].predict([Xquery])

