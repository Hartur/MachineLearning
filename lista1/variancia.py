import numpy

def calcularVar(X):
	soma = 0
	print(X)
	X = numpy.ones(len(X)) - X
	print(X)

	for x in X:
		soma = soma + 1 - x

	media = soma / len(X)
	print("media: " + format(media, '.3f'))
	print("len(X): " + str(len(X)))

	variancia = 0

	for x in X:
		variancia = variancia + numpy.power((x - media), 2)
		print("Somatório depois de " + str(x) + " = " + format(variancia, '.3f'))

	variancia = variancia / len(X)
	print("Variancia = " + format(variancia, '.3f'))

calcularVar([0.02,0.08,0.19,0.29,0.26,0.13,0.03])