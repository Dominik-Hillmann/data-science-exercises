import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

# get the data like before
(rawTrainImgs, rawTrainLbls), (rawTestImgs, rawTestLbls) = mnist.load_data()

# we're not going to flatten data because CNNs can process them as the 2D data they are
# instead shape into width x length x #colChannels
from keras import backend as K
# this deals with possibility that there is a different order of length and color channels, etc.
if K.image_data_format() == 'channels_first':
	# channel is first element in vector
    trainImgs = rawTrainImgs.reshape(rawTrainImgs.shape[0], 1, 28, 28)
    testImgs = rawTestImgs.reshape(rawTestImgs.shape[0], 1, 28, 28)
    inputShape = (1, 28, 28)
else:
	# channel is last element in vector
    trainImgs = rawTrainImgs.reshape(rawTrainImgs.shape[0], 28, 28, 1)
    testImgs = rawTestImgs.reshape(rawTestImgs.shape[0], 28, 28, 1)
    inputShape = (28, 28, 1)
# 28 px by 28px image with 1 color channel
    
# normalize the encoding of each px to 0 to 1
trainImgs = trainImgs.astype('float32')
testImgs = testImgs.astype('float32')
trainImgs /= 255
testImgs /= 255

# convert the labels 0 ... 9 to one hot encoding
trainLbls = keras.utils.to_categorical(rawTrainLbls, 10)
testLbls = keras.utils.to_categorical(rawTestLbls, 10)


# now we model the CNN
# you should use preconfigured ones normally since there are so many dials to turn
model = Sequential()
model.add(Conv2D(
	32, 						# 32 regional samples from the image
	kernel_size = (3, 3),		# each one of those samples has a 3 x 3 px kernel size 
    activation = 'relu',
    input_shape = inputShape
))

model.add(Conv2D(				# another Conv2D to make out higher level patterns
	64, 						# again subsampling of 3 by 3 px in 64 neurons
	(3, 3), 
	activation = 'relu'
))

model.add(MaxPooling2D(pool_size = (2, 2))) # Reduce by taking the max of each 2x2 block

model.add(Dropout(0.25)) 		# Dropout to avoid overfitting

model.add(Flatten())			# Flatten the results to one dimension for passing into our final layer
								# from here on this is a normal perceptron
model.add(Dense(128, activation = 'relu')) # hidden layer

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax')) # Final categorization from 0-9 with softmax

model.summary()

model.compile(
	loss = 'categorical_crossentropy', 	# appropriate loss fn for multiple categories porblem
	optimizer = 'adam',					
	metrics = ['accuracy']
)

history = model.fit(
	trainImgs, trainLbls,
	batch_size = 32,
	epochs = 10,
	verbose = 2,
	validation_data = (testImgs, testLbls)
)

score = model.evaluate(testImgs, testLbls, verbose = 2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
