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
	# there is also a KerasClassifier and KerasRegressor interface
	build_fn = createModel,
	epochs,
	verbose = 0
)

crossValScores = cross_val_scorce(estimator, features, labels, cv = 10)
print(crossValScores, crossValScores.mean())

"""
***** Convolutional Neural Networks CNNs *****
> what are they used for?
	> when you want to search for pattern but there is not any specific spot to search for them
	> e.g. images with certain features, machine translation, sentence classification, sentiment/mood analysis
	> like signs in a pic or words within a sentence
> how do they work? 
	> inspired by how brain processes images
	> subsampling: certain neurons receive only a certain part of your data, a part of your visual field
	> overlap to create the entire field of vision "convolution"
	> convolution is a fancy word for breaking data up in parts and process chunks individually
	> the first neurons might identify some simple structure like a line
	> they feed into higher layers identifying more and more complex structures 
	> when e.g. one neuron discovers a line and certain other neurons too, this next-layer-neuron probably fires bc it found a structure/pattern
> in color, this needs to be done 3 times: red, blue, green

> CNNs with Keras
	> source data into appropriate dimensions
	> Conv2D (1D, 3D, not necessarily image data) does the convolution
	> MaxPooling2D layers reduce pictures by taking max val of a block - way to shrinken images
		> to reduce computation - useful to reduce computation
	> at some point fed into Perceptron -> data has to be flattened and processed further on
	> typical:
	  Conv2D - MAxPooling2D - Dropout - Flatten - Dense - Dropout - Softmax
	> actual magic at lower levels

> very intensive on CPU, GPU, RAM
> lots of hyperparameters, like amount of pooling, layers, choice of optimizer

> already a lot of research and there a topologies that have proven to be most useful for certain problems:
	> LeNet-5 (handwriting), AlexNet (image classification), ResNet ("skip connections" between non neighbouring layers)
"""

"""
***** Recurrent Neural Networks RNNs *****
> for Time-Series data
> predict future behvior based on past behavior
> e.g. web logs, sensor logs, stock trades
> data that consists of sequences of arbitrary length

> how does an individual neural work?
	> like a normal neuron, but the output of the neuron from the past run get fed into the neuron again
	> "memory cell" - it remembers its past state
	> often adjustments: the more recent a behavior the more influence on the current state
> topologies
	> enables us to deal with sequences, not just snapshots

	> sequence to sequence: predict stock prices based on past prices
	> seq to vector: sentence --> sentiment
	> vec to seq: image --> image captions
	> encoder --> decoder
		> RNNs feeding into each other, e.g. Chinese sentence --> meaning --> German sentence
		> would look like seq --> vec --> seq, translation
> training
	> backpropagation through time
		> not only through network but through each past point in time
		> adds up very fast
		> upper cap to limit: to certain # steps back in time
> counteract effect that data from distant points in time matters less
	> because data keeps being fed
	> you don't want to give preference to more recent data: LSTM cells and GRU cells
> very sensitive to topologies and hyperparameters, resource intesive
	> wrong choice can lead RNN to not converge really fast
	
"""
