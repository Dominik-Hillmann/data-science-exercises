"""
***** Keras *****
> a higher level API to do NN than pure Tensorflow
> comes with Tensorflow
> Keras takes care of a lot of details but is a bit slower because of this
> it is good for prototyping, the fine tuning can be done with pure tensorflow
> more time spent on tuning and figuring out a good topology for the NN
> it is integrated with SciKit Learn
"""

"""
***** Types of problems for NNs *****
> multi-class classification: like MNIST where classification can be 0 to 9 (10 possible classifications)
	> Keras suggests to start to solve this problem like in (1)
> binary classification: decipher sex from images for example
	> suggestion in (2)
"""

# (1) Keras suggestion multi-class classification starting point
model = Sequential()

model.add(Dense(64, activation = 'relu', input_dim = 20))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(
	lr = 0.01,
	decay = 1e-6,
	momentum = 0.9,
	nesterov = True
) # SGD-optimizer

model.compile(
	loss = 'categorical_crossentropy',
	optimizer = sgd,
	metrics = ['accuracy']
)


# (2) starting point binary problems
model = Sequential()

model.add(Dense(64, activation = 'relu', input_dim = 20))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid')) # 0 or 1 output, activated by sigmoid

model.compile(
	loss = 'binary_crossentropy',
	optimizer = 'rmsprop',
	metrics = ['accuracy']
)


# snippet for K-Fold cross validation with SciKit Learn
def createModel():
	# ...
	# something like in (1) or (2)
	# ...
	return model

from tensorflow.Keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(
	build_fn = createModel,
	epochs,
	verbose = 0
)

crossValScores = cross_val_scorce(estimator, features, labels, cv = 10)
print(crossValScores, crossValScores.mean())