import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # handwriting data

session = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
for i in range(3):
    print()
# one hot encoding: better for neural Networks, array of 10 where one of them == 1, else 0 to mark which number was written
# e.g. 3 == [0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0]

# MNIST has 55000 images of written number, each 28 by 28 px = 783 px, this is a training data something
# then there's a test data set of 10000 images
# shape of the training data: 55000 observations with 784 variables each
# shape of test data: 55000 observations with 10 dimensions for one_hot for numbers 0-9

# Let's view a sample of this:
import matplotlib.pyplot as plot
def displaySample(num):
    print(mnist.train.labels[num]) # one_hot array of observation num
    label = mnist.train.labels[num].argmax(axis = 0) # convert one_hot encoding back to normal number
    image = mnist.train.images[num].reshape([28, 28]) # reshape array of values back to a 28x28 pic

    plot.title("Sample: %d Label: %d" % (num, label)) # pyplot, give it its legend
    plot.imshow(image, cmap = plot.get_cmap("gray_r"))
    plot.show()

import random
displaySample(random.randrange(1, 55000))

# don't forget: the neural network will "see" the flattened 1D array of the 784 values

# now we start to use Tensorflow
# first some placeholders for the images and labels of the data
inputImgs = tf.placeholder(tf.float32, shape = [None, 784]) # this says that there are an arbitrary (None) number of images with 784 values which are 32 bit floating point numbers
inputImgs = tf.placeholder(tf.float32, shape = [None, 10]) # arbitrary number of one_hot arrays of 10 values

# topology of the neural network
numHiddenNodes = 512

# weights and biases from input to hidden: 784 (image as 1D array in), 512 out to hidden layer
inputWeights = tf.Variable(tf.truncated_normal([784, numHiddenNodes])) # hidden layer: 784 inputs from first layer and 512 outputs
inputBiases = tf.Variable(tf.zeros([numHiddenNodes])) # biased terms in this layer

numOutNodes = 10 
# because we want to know which of the 10 numbers it is, one_hot encoding as output layer (that's why)

# now the biases and weights from the hidden layer, 512 to the output layer, 10 (describes hidden layer directly)
hiddenWeights = tf.Variable(tf.truncated_normal([numHiddenNodes, numOutNodes])) # truncated_normal normalizes
hiddenBiases = tf.Variable(tf.zeros([numOutNodes]))





















