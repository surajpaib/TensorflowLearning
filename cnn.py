import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print mnist

sess = tf.InteractiveSession()

width = 28
height = 28
flat = width * height
output = 10

# Input and Output placeholders
x = tf.placeholder(tf.float32, [None, flat])
y_ = tf.placeholder(tf.float32, [None, output])

x_image = tf.reshape(x, [-1, 28, 28, 1])


# Layer1 - Conv Layer
# the shape 5, 5, 1, 32 refers to 5 x 5 kernel, 1 channel input and 32 channel output
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# the shape 1, 1, 1, 1 represents batch, height, width, channel strides respectively
conv1_op = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
conv1_relu = tf.nn.relu(conv1_op)

# Layer2 - Max Pooling
conv1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Layer3 - Conv Layer
# the shape 5, 5, 1, 32 refers to 5 x 5 kernel, 1 channel input and 32 channel output
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# the shape 1, 1, 1, 1 represents batch, height, width, channel strides respectively
conv2_op = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
conv2_relu = tf.nn.relu(conv1_op)

# Layer2 - Max Pooling
conv1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

