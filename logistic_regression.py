import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Load the dataset
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
print trainX[0], trainY[0]

# Get shape of input and output variables
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

# Set placeholders training examples
x = tf.placeholder(tf.float32, [None, numFeatures])
y = tf.placeholder(tf.float32, [None, numLabels])

# Set weight and bias variables
# W = tf.Variable(tf.zeros([4, 3]))
# b = tf.Variable(tf.zeros([3]))

weights = tf.Variable(np.random.rand(4, 3).astype(np.float32))
bias = tf.Variable(np.random.rand(3).astype(np.float32))
print weights, bias

Wx = tf.matmul(x, weights)
Wxb = tf.add(Wx, bias)
Activation = tf.nn.sigmoid(Wxb)

learningRate = tf.train.exponential_decay(learning_rate=0.004,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

mse = tf.nn.l2_loss(y - Activation, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learningRate)
loss = optimizer.minimize(mse)
init = tf.global_variables_initializer()

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Activation, 1), tf.argmax(y, 1)), "float"))
iterations = 10000
with tf.Session() as sess:
    sess.run(init)
    for i in range(iterations):
        loss_rate = sess.run(loss, feed_dict={x: trainX, y: trainY})
        if i % 50 == 0:
            accuracy_val = sess.run(accuracy, feed_dict={x: trainX, y: trainY})
            mse_error = sess.run(mse, feed_dict={x: trainX, y: trainY})
            print "Accuracy: " + str(accuracy_val * 100) + " Loss: " + str(mse_error)
    test_accuracy = sess.run(accuracy, feed_dict={x: testX, y: testY})
    print "Test Set Accuracy:" + str(test_accuracy * 100)




