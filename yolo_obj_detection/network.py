import tensorflow as tf
import logging
import numpy as np
import itertools
import cv2
import voc_utils
import voc_train

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
classes = []

class YOLO_Network:
    def __init__(self, alpha, lambda_coord, lambda_noobj, trainable, total_labels):
        self.alpha = alpha
        self.labels = total_labels
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.trainable = trainable
        self.index_in_epoch = 0

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
        self.fc_3 = self.FC_Layer(18, self.dropout, 7 * 7 * self.labels, flat=False, trainable=self.trainable)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def training_metrics(self):
        # Training side output information
        # YOLO model splits the input into 7 x 7 grid and predicts two bounding boxes per each grid hence the [7, 7, 2] tensor for each value. obj indicates presence of an object in one unit of the grid for each
        # of the bounding boxes, similarly noobj as well. objI indicates if any object is present in one unit and is a [7, 7] tensor

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

        # Among the final 1470 size output vector first 7x7x self.labels elements represent class probabilities for number of labels

        class_probs = tf.reshape(output[0:nb_image, 0: (7 * 7 * self.labels)], (nb_image, 7, 7, self.labels))
        scales = tf.reshape(output[0:nb_image, (7 * 7 * self.labels): (7 * 7 * self.labels) + 7 * 7 * 2], (nb_image, 7, 7, 2))
        boxes = tf.reshape(output[0:nb_image, (7 * 7 * self.labels) + 7 * 7 * 2:], (nb_image, 7, 7, 2, 4))
        boxes0 = boxes[:, :, :, :, 0]
        boxes1 = boxes[:, :, :, :, 1]
        boxes2 = boxes[:, :, :, :, 2]
        boxes3 = boxes[:, :, :, :, 3]

        ## Loss Functions for each of the above metrics
        self.subX = tf.subtract(boxes0, self.x_)
        self.subY = tf.subtract(boxes1, self.y_)
        self.subW = tf.subtract(boxes2, self.w_)
        self.subH = tf.subtract(boxes3, self.h_)
        self.subC = tf.subtract(scales, self.C_)
        self.subP = tf.subtract(class_probs, self.p_)

        # Each loss is a squared difference between the actual and the predictions. It is multiplied with self.obj to consider the metric only if the object is present and then multiplied
        # with self.lambda_coord which is a weighted value.
        self.lossX = tf.multiply(self.lambda_coord, tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subX, self.subX)), axis=[1, 2, 3]))
        self.lossY = tf.multiply(self.lambda_coord, tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subY, self.subY)), axis=[1, 2, 3]))
        self.lossW = tf.multiply(self.lambda_coord, tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subW, self.subW)), axis=[1, 2, 3]))
        self.lossH = tf.multiply(self.lambda_coord, tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subH, self.subH)), axis=[1, 2, 3]))
        self.lossC_Obj = tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subC, self.subC)), axis=[1, 2, 3])
        self.lossC_Noobbj = tf.multiply(self.lambda_noobj, tf.reduce_sum(tf.multiply(self.noobj, tf.multiply(self.subC, self.subC)), axis=[1, 2, 3]))
        self.lossP = tf.reduce_sum(tf.multiply(self.objI, tf.reduce_sum(tf.multiply(self.subP, self.subP), axis=3)), axis=[1, 2])
        # Combining the loss functions to generate a scalar loss metric
        self.loss = tf.add_n((self.lossX, self.lossY, self.lossW, self.lossH, self.lossP, self.lossC_Obj, self.lossC_Noobbj))
        self.loss = tf.reduce_mean(self.loss)

        logger.info("Total Combined loss for all outputs in the network: {0}".format(type(self.loss)))

        global_step = tf.Variable(0, trainable=False)
        starter_lr = 0.004
        self.epoch = tf.placeholder(tf.int32)

        self.train_step = tf.train.MomentumOptimizer(learning_rate=starter_lr, momentum=0.9).minimize(self.loss, global_step=global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_label(self, img_files, epoch):
        X_global = []
        Y_global = []
        W_global = []
        H_global = []
        C_global = []
        P_global = []
        obj_global = []
        objI_global = []
        noobj_global = []
        Image = []
        for img_file in img_files:
            prelabel = voc_train.get_training_data(img_file)
            x = np.zeros([7, 7, 2])
            x = np.zeros([7, 7, 2])
            y = np.zeros([7, 7, 2])
            w = np.zeros([7, 7, 2])
            h = np.zeros([7, 7, 2])
            C = np.zeros([7, 7, 2])
            p = np.zeros([7, 7, 20])
            obj = np.zeros([7, 7, 2])
            objI = np.zeros([7, 7])
            noobj = np.ones([7, 7, 2])
            img = voc_utils.load_img(img_file)

            for i, j in itertools.product([0, 7], [0, 7]):
                if prelabel[i][j] is not None:
                    index = 0
                    while (len(prelabel[i][j]) > index and index < 2):
                        x[i][j][index] = (float(prelabel[i][j][index][0])/len(img)) * 7 - i
                        y[i][j][index] = (float(prelabel[i][j][index][1])/len(img)) * 7 - j
                        w[i][j][index] = np.sqrt(prelabel[i][j][index][2]) / len(img) * 7
                        h[i][j][index] = np.sqrt(prelabel[i][j][index][2]) / len(img[0])
                        C[i][j][index] = 1.0
                        p[i][j][self.classes.index(prelabel[i][j][index][4])] = 1.0 / float(len(prelabel[i][j]))
                        obj[i][j][index] = 1.0
                        objI[i][j] = 1.0
                        noobj[i][j][index] = 0.0
                        index = index + 1
            X_global.append(x)
            Y_global.append(y)
            W_global.append(w)
            H_global.append(h)
            C_global.append(C)
            P_global.append(p)
            obj_global.append(obj)
            objI_global.append(objI)
            noobj_global.append(noobj)

            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray(img_RGB)
            inputs = np.zeros((1, 448, 448, 3), dtype='float32')
            inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
            Image.append(inputs[0])
            X_global = np.array(X_global)
            Y_global = np.array(Y_global)
            W_global = np.array(W_global)
            H_global = np.array(H_global)
            C_global = np.array(C_global)
            P_global = np.array(P_global)
            obj_global = np.array(obj_global)
            objI_global = np.array(objI_global)
            noobj_global = np.array(noobj_global)
            Image = np.array(Image)

        return {self.x: Image, self.x_: X_global, self.y_: Y_global, self.w_: W_global, self.h_: H_global,
                    self.C_: C_global,
                    self.p_: P_global, self.obj: obj_global, self.objI: objI_global, self.noobj: noobj_global, self.epoch: epoch}

    def next_batch(self, batch_size, num_examples):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > num_examples:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            self.label = self.label[perm]
            start = 0
            self.index_in_epoch = batch_size
            assert  batch_size <= num_examples
        end = self.index_in_epoch
        return self.label[start: end]






    def training_step(self, i, update_test, update_train):
        for nbatch in range(0, len(self.label)/64):
            train_dict = self.build_label(self.next_batch(12, num_examples=len(self.label)), i)
            self.sess.run(self.train_step, train_dict)


        train_l = []
        test_l = []

        if update_train:
            l = self.sess.run(self.loss, feed_dict=self.build_label(self.label, i))
            logger.info('Training Loss: {0}'.format(l))
            train_l.append(l)


        if update_test:
            l = self.sess.run(self.loss, feed_dict=self.build_label(self.label_test_, i))
            test_l.append(l)
            test_l.append(l)



        return (train_l, test_l)

    def train(self):
        train_l = []
        test_l = []
        self.label = voc_utils.imgs_from_category_as_list("bird", "train")
        self.label_test = voc_utils.imgs_from_category_as_list("bird", "val")
        training_iter = 137
        epoch_size = 5
        for i in range(training_iter):
            test = False
            if i % epoch_size == 0 and i != 0:
                test = True
            l, tl = self.training_step(i, test, test)
            train_l += l
            test_l += tl
        print("train loss")
        print(train_l)
        print("test loss")
        print(test_l)


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
    net = YOLO_Network(alpha=0.1, lambda_coord= 5.0, lambda_noobj=0.5, trainable= True, total_labels=20)
    net.define_network()
    net.training_metrics()
