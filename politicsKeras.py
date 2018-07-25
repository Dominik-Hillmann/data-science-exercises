# try to predict whether a congressman is Republican or Democrat from how they voted on 17 issues
# older data (pre 1990s)

import pandas as pd

featureNames = [
	'party','handicapped-infants', 'water-project-cost-sharing', 
    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
    'el-salvador-aid', 'religious-groups-in-schools',
    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
    'mx-missle', 'immigration', 'synfuels-corporation-cutback',
    'education-spending', 'superfund-right-to-sue', 'crime',
    'duty-free-exports', 'export-administration-act-south-africa'
] # names of the 17 issues

votingData = pd.read_csv(
	'./datasets/house-votes-84.data.txt', # name file
	na_values = ['?'], 		   			  # how missing values are marked
    names = featureNames	   			  # names of the colums == feature_names
)

print(votingData.head())
print(votingData.describe())
# rows are congressmen, columns are issues with yes, party, no and missing values

# we just drop the rows with missing values
# should not be done so easily normally because you could introduce hidden bias here
votingData.dropna(inplace = True)
print(votingData.describe()) # there are only 232 congressmen left without missing votes

# now we prepare the data so that neural network digests it, only numerical data!
votingData.replace(('y', 'n'), (1, 0), inplace = True) # replace yes and no with 1 and 0
votingData.replace(('democrat', 'republican'), (1, 0), inplace = True)

voteBehavior = votingData[featureNames].drop('party', axis = 1).values # whole matrix without parties
party = votingData['party'].values # only vector party
print('Same length:', len(party) == len(voteBehavior))

# we're done massaging the data, now create the model:
# create model that predicts the party
# also use K-Fold cross validation from sci-kit learn!

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

def randTestData(Y, X, num):
	from random import randint
	drawn = []
	testY = []
	testX = []
	length = len(Y)

	while len(testY) < num:
		# print(len(Y), len(X), len(testX), len(testY))
		length = len(Y)
		i = randint(0, length - 1)

		while i in drawn: i = randint(0, length - 1)

		drawn.append(i)

		testY.append(Y[i])
		testX.append(X[i])

		Y = np.delete(Y, i)
		X = np.delete(X, i, 0)

	return (np.array(testY), np.array(testX), Y, X)

(testParty, testBehavior, party, voteBehavior) = randTestData(
	party, 
	voteBehavior, 
	int(0.1 * len(party))
)


# shape
print("Training data:", (len(party), len(voteBehavior), len(voteBehavior[0])))
print("Test data:", (len(testParty), len(testBehavior), len(testBehavior[0])))

def Model():
	model = Sequential()

	# first hidden layer: 64 neurons with shape (#dimensions, #observations), ReLU activation fn
	model.add(Dense(
		64,
		activation = 'relu',
		input_shape = (len(voteBehavior[0]), )
	))
	model.add(Dropout(0.5))

	# second hidden layer with dropout
	model.add(Dense(32, activation = 'relu'))
	model.add(Dropout(0.5))

	model.add(Dense(16, activation = 'relu'))
	model.add(Dropout(0.5))	

	# output layer, only yes or no: Democrat?
	model.add(Dense(1, activation = 'sigmoid')) 

	model.compile(
		loss = 'binary_crossentropy',
		optimizer = 'rmsprop',
		metrics = ['accuracy']
	)

	return model

model = Model()
print(model.summary())

history = model.fit(
	voteBehavior, party,
	batch_size = 2,
	epochs = 100,
	verbose = 2,
	validation_data = (testBehavior, testParty)
)

print(model.evaluate(testBehavior, testParty, verbose = 2))

# from tensorflow.Keras.wrappers.scikit_learn import KerasClassifier
# evaluate using 10-fold cross validation
estimator = KerasClassifier(
	build_fn = Model,
	nb_epoch = 200,
	batch_size = 1,
	verbose = 2
)

from sklearn.model_selection import cross_val_score
results = cross_val_score(
	estimator, 
	voteBehavior, 
	party,
	cv = 3 # k, number of folds
)
print(results.mean())