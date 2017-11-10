import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data



class CNN:
    def __init__(self):

        self.sess = tf.InteractiveSession()

        width = 28
        height = 28
        flat = width * height
        output = 10

        self.x = tf.placeholder(tf.float32, [None, flat])
        self.y_ = tf.placeholder(tf.float32, [None, output])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    def layer1_ConvMaxPool(self, input):
        """
        Conv2D Layer + Max Pooling
        :return:
        """

        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

        conv1_op = tf.nn.conv2d(input, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        conv1_relu = tf.nn.relu(conv1_op)

        conv1 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return conv1

    def layer2_COnvMaxPool(self, input):
        """
        Conv2D + Max Pool
        :param input:
        :return:
        """
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        conv2_op = tf.nn.conv2d(input, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        conv2_relu = tf.nn.relu(conv2_op)

        conv2 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return conv2

    def layer3_flatten(self, input):
        """
        Flatten 2D layers to 1D
        :param input:
        :return:
        """
        flatten = tf.reshape(input, [-1, 7 * 7 * 64])

        return flatten


    def layer4_fullyconnected(self, input):
        """
        Fully connected Layer
        :param input:
        :return:
        """
        W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        fc_layer = tf.matmul(input, W_fc1) + b_fc1

        fc_out = tf.nn.relu(fc_layer)

        return fc_out

    def layer5_dropout(self, input):
        """
        Dropout + Fully Connected
        :return:
        """

        self.dropout_prob = tf.placeholder(tf.float32)
        do_layer = tf.nn.dropout(input, keep_prob=self.dropout_prob)

        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

        o_layer = tf.matmul(do_layer, W_fc2) + b_fc2
        output = tf.nn.softmax(o_layer)
        return output


    def define_metrics(self):

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_pred), axis=[1]))
        self.training_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.correct_pred = tf.equal(tf.argmax(self.y_, 1), tf.arg_max(self.y_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))



    def fit(self, X, Y, iterations=200):
        layer1_out = self.layer1_ConvMaxPool(self.x_image)
        layer2_out = self.layer2_COnvMaxPool(layer1_out)
        layer3_out = self.layer3_flatten(layer2_out)
        layer4_out = self.layer4_fullyconnected(layer3_out)
        self.y_pred = self.layer5_dropout(layer4_out)

        self.define_metrics()


        self.sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            self.training_step.run(feed_dict={self.x: X, self.y_: Y, self.dropout_prob: 0.5})

            if i % 100 == 0:
                acc_train = self.accuracy.eval(feed_dict={self.x: X, self.y_: Y, self.dropout_prob: 1.0})
                print "Training Accuracy: {0}".format(acc_train * 100)

    def score(self, X, Y):
        test_acc = self.accuracy.eval(feed_dict={self.x: X, self.y_: Y, self.dropout_prob:1.0})
        print "Test Images Accuracy: {0}".format(test_acc * 100)

    def predict(self, X):
        out = self.y_pred.eval(feed_dict={self.x: X, self.dropout_prob:1.0})
        return out

if __name__ == "__main__":
    cnn_classifier = CNN()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch = mnist.train.next_batch(1)
    cnn_classifier.fit(batch[0], batch[1])
    print cnn_classifier.predict(batch[0])