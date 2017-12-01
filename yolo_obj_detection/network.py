import tensorflow as tf
import logging
import numpy as np

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class YOLO_Network:
    def __init__(self, alpha, trainable):
        self.alpha = alpha
        self.trainable = trainable

    def define_network(self):
        logger.info('Building Network for Yolo')
        self.x = tf.placeholder(tf.float32, [None, 448, 448, 3])
        self.conv1 = self.Convolution2D_Layer(1, self.x, 64, 7, 2)
        self.max_pool1 = self.MaxPool_Layer(2, self.conv1, 2, 2)
        self.conv2 = self.Convolution2D_Layer(3, self.max_pool1, 192, 3, 1)
        self.pool2 = self.MaxPool_Layer(4, self.conv2, 2, 2)
        self.conv3 = self.Convolution2D_Layer(5, self.pool2, 128, 1, 1)
        self.conv4 = self.Convolution2D_Layer(6, self.conv3, 256, 3, 1)
        self.conv5 = self.Convolution2D_Layer(7, self.conv4, 256, 1, 1)
        self.conv6 = self.Convolution2D_Layer(8, self.conv5, 512, 3, 1)
        self.pool3 = self.MaxPool_Layer(9, self.conv6, 2, 2)
        self.conv7 = self.Convolution2D_Layer(10, self.pool3, 512, 1, 1)
        self.conv8 = self.Convolution2D_Layer(11, self.conv7, 1024, 1, 1)
        self.conv9 = self.Convolution2D_Layer(12, self.conv8, 1024, 3, 1)
        self.pool4 = self.MaxPool_Layer(13, self.conv9, 2, 2)
        self.conv10 = self.Convolution2D_Layer(14, self.pool4, 1024, 1, 1)
        self.pool5 = self.MaxPool_Layer(15, self.conv10, 2, 2)
        self.conv11 = self.Convolution2D_Layer(16, self.pool5, 512, 3, 1)
        self.pool6 = self.MaxPool_Layer(17, self.conv11, 2, 2)
        self.conv12 = self.Convolution2D_Layer(14, self.pool6, 1024, 3, 1)
        self.fc_1 = self.FC_Layer(15, self.conv12, 512, flat=True, trainable=self.trainable)
        self.fc_2 = self.FC_Layer(16, self.fc_1, 4096, flat=False, trainable=self.trainable)
        self.dropout = self.Dropout_Layer(17, self.fc_2, 0.5)
        self.fc_3 = self.FC_Layer(18, self.dropout, 1470, flat=False, trainable=self.trainable)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def training_metrics(self):
        self.x_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.y_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.w_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.h_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.C_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.p_ = tf.placeholder(tf.float32, [None, 7, 7, 20])
        self.obj = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.objI = tf.placeholder(tf.float32, [None, 7, 7])
        self.noobj = tf.placeholder(tf.float32, [None,7, 7, 2])

        output = self.fc_3
        nb_image = tf.shape(self.x_)[0]
        class_probs = tf.reshape(output[0:nb_image, 0:980], (nb_image, 7, 7, 20))
        scales = tf.reshape(output[0:nb_image, 980:1078], (nb_image, 7, 7, 2))
        boxes = tf.reshape(output[0:nb_image, 1078:], (nb_image, 7, 7, 2, 4))
        boxes0 = boxes[:, :, :, :, 0]
        boxes1 = boxes[:, :, :, :, 1]
        boxes2 = boxes[:, :, :, :, 2]
        boxes3 = boxes[:, :, :, :, 3]

        self.subX = tf.subtract(boxes0, self.x_)
        self.subY = tf.subtract(boxes1, self.y_)
        self.subW = tf.subtract(boxes2, self.w_)
        self.subH = tf.subtract(boxes3, self.h_)
        self.subC = tf.subtract(scales, self.C_)
        self.subP = tf.subtract(class_probs, self.p_)
        self.loss = tf.multiply(self.lambda_c, tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subX, self.subX))))





    def Convolution2D_Layer(self, layer_idx, input, filter, size, stride, trainable=False):

        channels = input.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filter], stddev=0.1), trainable=trainable)
        bias = tf.Variable(tf.constant(0.1, shape=[filter]), trainable=trainable)
        #
        # pad_size = size // 2
        # pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        # padded_input = tf.pad(input, pad_mat)
        conv = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME', name=str(layer_idx)+'_conv')
        conv_bias = tf.add(conv, bias)
        logger.info('Layer {0}: Type: Convolution Stride: {1} Filter: {2}, Input Shape: {3}'.format(layer_idx, stride, str([size, size, int(channels), filter]), str(input.get_shape())))
        return tf.maximum(tf.multiply(self.alpha, conv_bias), conv_bias)

    def MaxPool_Layer(self, layer_idx, input, size, stride):
        channels = input.get_shape()[3]
        logger.info('Layer {0}: Type: Pooling Stride: {1} Filter: {2}, Input Shape: {3}'.format(layer_idx, stride, str([size, size, int(channels), int(input.get_shape()[1]/ size)]), str(input.get_shape())))
        return tf.nn.max_pool(input, strides=[1, stride, stride, 1], ksize=[1, size, size, 1], padding='SAME')

    def FC_Layer(self, layer_idx, input, hidden, flat=False, linear=False, trainable=False):
        input_shape = input.get_shape().as_list()
        if flat:
            dim = input_shape[1] *input_shape[2] * input_shape[3]
            input_T = tf.transpose(input, (0, 3, 1, 2))
            input_processed = tf.reshape(input_T, [-1, dim])
        else:
            dim = input_shape[1]
            input_processed = input

        weight = tf.Variable(tf.zeros([dim, hidden]), trainable=trainable)
        bias = tf.Variable(tf.constant(0.1, shape=[hidden]), trainable=trainable)
        ip = tf.add(tf.matmul(input_processed, weight), bias)
        logger.info('Layer {0}: Type: Fully Connected Output:{1} Input Shape: {2}'.format(layer_idx, hidden, str(input.get_shape())))
        return tf.maximum(tf.multiply(self.alpha, ip), ip)

    def Dropout_Layer(self, layer_idx, input, dropput_prob):
        logger.info('Layer {0}: Type: Dropout Rate:{1} Input Shape: {2}'.format(layer_idx, dropput_prob, str(input.get_shape())))
        return tf.nn.dropout(input, keep_prob=dropput_prob)

if __name__ == "__main__":
    net = YOLO_Network(alpha=0.1, trainable= True)
    net.define_network()