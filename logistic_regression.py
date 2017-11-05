import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.004):
        self.lr = lr

    def _construct_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.numFeatures])
        self.y = tf.placeholder(tf.float32, [None, self.numLabels])
        self.weights = tf.Variable(np.random.rand(self.numFeatures, self.numLabels).astype(np.float32))
        self.bias = tf.Variable(np.random.rand(self.numLabels).astype(np.float32))
        Wx = tf.matmul(self.x, self.weights)
        Wxb = tf.add(Wx, self.bias)
        Activation = tf.nn.sigmoid(Wxb)
        learningRate = tf.train.exponential_decay(learning_rate=self.lr,
                                                  global_step=1,
                                                  decay_steps=trainX.shape[0],
                                                  decay_rate=0.95,
                                                  staircase=True)

        self.mse = tf.nn.l2_loss(self.y - Activation, name='loss')
        self.optimizer = tf.train.GradientDescentOptimizer(learningRate)
        self.loss = self.optimizer.minimize(self.mse)
        self.init = tf.global_variables_initializer()
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Activation, 1), tf.argmax(self.y, 1)), "float"))

    def fit(self, X, Y):
        self.numFeatures = X.shape[1]
        self.numLabels= Y.shape[1]
        self._construct_graph()

        iterations = 10000
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(iterations):
                sess.run(self.loss, feed_dict={self.x: X, self.y: Y})
                if i % 50 == 0:
                    accuracy_val = sess.run(self.accuracy, feed_dict={self.x: X, self.y: Y})
                    mse_error = sess.run(self.mse, feed_dict={self.x: X, self.y: Y})
                    print "Accuracy: " + str(accuracy_val * 100) + " Loss: " + str(mse_error)

    def predict(self, X, Y):
        with tf.Session() as sess:
            sess.run(self.init)
            accuracy_val = sess.run(self.accuracy, feed_dict={self.x: X, self.y: Y})
            print accuracy_val


if __name__ == "__main__":
    iris = load_iris()
    iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
    iris_y= pd.get_dummies(iris_y).values
    trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
    print trainX[0], trainY[0]

    # Get shape of input and output variables
    logregressor = LogisticRegression()
    logregressor.fit(trainX, trainY)

# Set placeholders training examples


