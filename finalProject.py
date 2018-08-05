"""
This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

	BI-RADS assessment: 1 to 5 (ordinal) (discarded)
	Age: patient's age in years (integer)
	Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
	Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
	Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
	Severity: benign=0 or malignant=1 (binominal)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# ***** Part 1: Data Preparation *****

colNames = ['BI_RADS', 'age', 'shape', 'margin', 'density', 'severity']
data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data', 
	na_values = ['?'],
	names = colNames
)
print(data.head())

print(data.describe())
print()
print(data.shape)
print()
# print(data[data[['age']] > 50.0].dropna().shape)
data = data.dropna()
print(data.head())
print(data.shape)

# create NumPy arrays for the features
Y = (data[['severity']].values) #.flatten() cannot be flattened here because standard sclaer expects 2D array
X = data.drop(['BI_RADS', 'severity'], axis = 1).values
print('X shape: ' + str(X.shape))
print('Y shape: ' + str(Y.shape))

colNames.remove('BI_RADS')
colNames.remove('severity')

# some models will need the data to be normalized
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# print(X)
# print(Y)

scaler.fit(X) # stores the mean and variance
normX = scaler.transform(X) # transforms data so that it fits to normal distribution of mean 0.0 and standard deviation 1.0

scaler.fit(Y)
normY = scaler.transform(Y)

# print(normX)
# print(normY)

def randTestData(Y, X, num):
	from random import randint
	drawn = []
	testY = []
	testX = []

	length = len(Y)
	i = randint(0, length - 1)

	while len(testY) < num:
		length = len(Y)

		while i in drawn: i = randint(0, length - 1)

		drawn.append(i)

		testY.append(Y[i])
		testX.append(X[i])

		Y = np.delete(Y, i)
		X = np.delete(X, i, 0)

	return (np.array(testY), np.array(testX), Y, X)

testY, testX, trainY, trainX = randTestData(np.array(Y), np.array(X), int(0.15 * len(Y)))

def correctPreds(predY, testY):
	corrects = 0
	dataLen = len(predY)
	for i in range(dataLen - 1):
		if testY[i] == predY[i]:
			corrects += 1
	return corrects / dataLen



# ***** 1st Model - Decesion Tree *****
def decesionTree(trainX, trainY, testX, testY, featureNames, useForest = False):
	if not useForest:
		from sklearn import tree
		classifier = tree.DecisionTreeClassifier()
	else:
		from sklearn.ensemble import RandomForestClassifier
		classifier = RandomForestClassifier(n_estimators = 3)

	classifier = classifier.fit(trainX, trainY)
	predY = classifier.predict(testX)

	print('Percentage of correctly predicted testYs using a Decesion ' + ('Forest' if useForest else 'Tree') + ': ' + str(correctPreds(predY, testY)) + '\n')

	return classifier
	
decesionTree(trainX, trainY, testX, testY, colNames)
decesionTree(trainX, trainY, testX, testY, colNames, useForest = True)
# both models usually achieve between 65% and 80% accuracy where the Forest always has a slight edge of ~3%



# ***** 2nd Model: Support Vector Machines *****
def applySVM(trainX, trainY, testX, testY, kernel):
	from sklearn import svm
	
	C = 1.0 # penalty parameter for error term
	classifier = svm.SVC(kernel = kernel, C = C).fit(trainX, trainY)

	predY =  classifier.predict(testX)

	print('Percentage of correctly predicted testYs using a Support Vector Machine with a ' + kernel + ' kernel: ' + str(correctPreds(predY, testY)) + '\n')

	return classifier

applySVM(trainX, trainY, testX, testY, 'linear') # linear kernel scores about as well as the decesion trees
applySVM(trainX, trainY, testX, testY, 'sigmoid') # about 50% correct predictions
applySVM(trainX, trainY, testX, testY, 'rbf')



# ***** 3rd Model: K Nearest Neighbours *****
def knn(trainX, trainY, testX, testY, k = 1):
	from sklearn.neighbors import KNeighborsClassifier

	classifier = KNeighborsClassifier(n_neighbors = k).fit(trainX, trainY)

	predY = classifier.predict(testX)
	print('Percentage of correctly predicted testYs using KNN with k = ' + str(k) + ': ' + str(correctPreds(predY, testY)))

	return classifier

for k in range(1, 50):
	knn(trainX, trainY, testX, testY, k = k)
	# k == 8 seems to work best with results of around 80% accuracy
print()



# ***** 4th Model: Logistic Regression *****
def logRegression(trainX, trainY, testX, testY):
	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression().fit(trainX, trainY)
	predY = classifier.predict(testX)

	print('Percentage of correctly predicted testYs using logistic regression: ' + str(correctPreds(predY, testY)))

	return classifier

logRegression(trainX, trainY, testX, testY)
# around 80% accuracy, too



# ***** 5th Model: Artificial Neural Network *****
def modelNN(trainX, trainY, testX, testY):
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout

	model = Sequential()

	model.add(Dense(
		64,
		activation = 'relu',
		input_shape = (len(trainX[0]), )
	))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation = 'sigmoid')) # 0 or 1 output, activated by sigmoid

	model.compile(
		loss = 'binary_crossentropy',
		optimizer = 'rmsprop',
		metrics = ['accuracy']
	)

	print(model.summary())

	classifier = model.fit(
		trainX, trainY,
		batch_size = 100,
		epochs = 830,
		verbose = 2,
		validation_data = (testX, testY)
	)

	# predY = classifier.predict(testY)
	# print('NN: ' + str(correctPreds(predY, testY)))

	return classifier

modelNN(trainX, trainY, testX, testY)
# the NN is able to predict with an accuracy of around 83%
# networks with more or less than 64 neurons in two layers perform about 10% worse
# networks with more layers perform 4% worse

#https://www.youtube.com/watch?v=YKP31T5LIXQ