import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print mnist

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Layer 1
W1 = tf.Variable(tf.zeros([784, 50], tf.float32))
b1 = tf.Variable(tf.zeros([50], tf.float32))
y_1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# Layer 2
W2 = tf.Variable(tf.zeros([50, 25], tf.float32))
b2 = tf.Variable(tf.zeros([25], tf.float32))
y_2 = tf.nn.sigmoid(tf.matmul(y_1, W2) + b2)

# Output Layer
W3 = tf.Variable(tf.zeros([25, 10], tf.float32))
b3 = tf.Variable(tf.zeros([10], tf.float32))
y = tf.nn.softmax(tf.matmul(y_2, W3) + b3)

sess.run(tf.global_variables_initializer())

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
train = tf.train.GradientDescentOptimizer(0.8)
entropy_loss = train.minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

for i in range(20000):
    batch = mnist.train.next_batch(50)
    entropy_loss.run(feed_dict={x:batch[0], y_: batch[1]})
    acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]}) * 100
    print ("Training Accuracy {0}".format(accuracy))



acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print (" Final Test accuracy : {0}".format(acc))