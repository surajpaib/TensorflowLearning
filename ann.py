import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print mnist

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))

sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
train = tf.train.GradientDescentOptimizer(0.5)
entropy_loss = train.minimize(cross_entropy)

for i in range(2000):
    batch = mnist.train.next_batch(50)
    entropy_loss.run(feed_dict={x:batch[0], y_: batch[1]})


correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print (" Final Test accuracy : {0}".format(acc))