import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # handwriting data something

session = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# one hot encoding: better for neural Networks
# print(mnist.shape)

import matplotlib.pyplot as pyplot
def display_sample(num):
    print(mnist.train.labels[num])
    label = mnist.train.labels[num].argmax(axis = 0)
    image = mnist.train.labels[num].reshape([28, 28])
    pyplot.title("Sampel: %d Label: %d" % (num, label))
    pyplot.imshow(image, cmap = pyplot.get_cmap("gray_r"))
    pyplot.show()
display_sample(123)
