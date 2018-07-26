# sentiment analysis: sequence to vector problem
# movie review sequence of words --> positive rating or not

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM # will need that because it doesn't matter where word in sentence appears
from keras.datasets import imdb # data set part of Keras

(trainX, trainY), (testX, testY) = imdb.load_data(num_words = 20000) # limit to occurences of 200000 most popular words

# take a look at what data looks like:
print("X:", trainX[0]) 	# array of words represented as numbers
						# e.g. 1 == "the", and these can range fro1 to 20000: 
						# 200000 most popular words
						# but the length of an X[i] is variable
print("Y", trainY[0])	# 0 or 1 <=> like it or not?

# this means that we only look at the first 80 words because RNNs blow up fast --> less computation
trainX = sequence.pad_sequences(trainX, maxlen = 80)
testX = sequence.pad_sequences(testX, maxlen = 80)

# we start to build the model!
model = Sequential()
model.add(Embedding(20000, 128))	# Embedding: converts data into vectors of fixed size
									# 200000 to match vocab size
									# 128 to match the num of neurons in the next layer: output of 128
model.add(LSTM(						# LSTM: so that data further from the past matters the same
	128,							# 128 LSTM cells
	dropout = 0.2, 					# dropout terms against overfitting
	recurrent_dropout = 0.2
))
model.add(Dense(1, activation = 'sigmoid')) # final neuron whether this is a positive sentiment or not


model.compile(
	loss = 'binary_crossentropy', 	# binary since only 0 or 1 output
	optimizer = 'adam',
	metrics = ['accuracy']
)

model.fit(
	# takes a very long time (> 1h)
	trainX, trainY,
	batch_size = 32,
	epochs = 15,
	verbose = 2,
	validation_data = (testX, testY)
)

score, acc = model.evaluate(
	testX, testY,
	batch_size = 32,
	verbose = 2
)
print('Test score:', score)
print('Test accuracy:', acc)