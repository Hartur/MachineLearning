from sklearn import datasets, neighbors, metrics, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

class Classifiers:
	KNN_ALGORITHM = 'brute'

	#KNN
	def getKNNClassifier(self, X, y, weights):
		numOfNeighbors = 15
		classifier = neighbors.KNeighborsClassifier(n_neighbors = numOfNeighbors, weights = weights, algorithm=self.KNN_ALGORITHM)
		classifier.fit(X, y)
		return classifier

	#SGD
	def getSGDClassifier(self, X, y):
		classifier = linear_model.SGDClassifier(loss="hinge", penalty="l2")
		classifier.fit(X, y)
		return classifier

	#Bayes
	def getBayesClassifier(self, X, y):
		classifier = GaussianNB()
		classifier.fit(X,y)
		return classifier

	#Decision Tree
	def getDecisionTreeClassifier(self, X, y):
		classifier = DecisionTreeClassifier(random_state=0)
		classifier.fit(X,y)
		return classifier