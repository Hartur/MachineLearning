import math

class DynamicSelectionUtils:
	def getDistance(self, x1, x2):
		result = 0
		for i in range(0, len(x1)):
			result += math.pow(x2[i] - x1[i], 2)
		return result
	
	def getNearst(self, n, X, y, Xquery):
		distances = []
		for i in range(len(X)):
			distances.append((X[i], y[i], self.getDistance(Xquery, X[i])))

		distances = sorted(distances, reverse=False, key=lambda d: d[2])
		nearest = distances[0:n]

		return nearest