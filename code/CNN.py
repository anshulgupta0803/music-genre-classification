
# coding: utf-8

# In[ ]:

from __future__ import print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from tflearn.data_utils import shuffle

class CNN(object):
    def __init__(self, patch_size, num_filters_fist_layer, num_filters_second_layer,
                 size_fully_connected_layer, image_x=400, image_y=400, image_channels=4, num_classes=10):
        
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.image_x = image_x
        self.image_y = image_y
        self.image_channels = image_channels
        image_size = self.image_x * self.image_y
        self.num_classes = num_classes
        
        self.x = tf.placeholder(tf.float32, shape=[None, image_x, image_y, image_channels])
        self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        def weight_variable(shape, nameVar):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=nameVar)

        def bias_variable(shape, nameVar):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=nameVar)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # First Layer (Convolution and Max Pool)
        self.W_conv1 = weight_variable([patch_size, patch_size, image_channels, num_filters_fist_layer], "filter_layer1")
        b_conv1 = bias_variable([num_filters_fist_layer], "bias_layer1")
        x_image = tf.reshape(self.x, [-1, image_x, image_y, image_channels])
        # Apply Convolution and Max Pool
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + b_conv1)
        print(h_conv1.get_shape())
        h_pool1 = max_pool_2x2(h_conv1)
        print(h_pool1.get_shape())

        # Second Layer (Convolution and Max Pool)
        self.W_conv2 = weight_variable([patch_size, patch_size, num_filters_fist_layer, num_filters_second_layer], "filter_layer2")
        b_conv2 = bias_variable([num_filters_second_layer], "bias_layer2")
        # Apply Convolution and Max Pool
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + b_conv2)
        print(h_conv2.get_shape())
        h_pool2 = max_pool_2x2(h_conv2)
        print(h_pool2.get_shape())

        # Fully Connected Layer
        W_fc1 = weight_variable([int(image_x / 4) * int(image_y / 4) * num_filters_second_layer, size_fully_connected_layer], "W_fc1")
        b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(image_x / 4) * int(image_y / 4) * num_filters_second_layer])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        print(h_fc1.get_shape())# the shape of h_fc1 is [-1, size_fully_connected_layer]

        # Add dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Add the last fully connected layer for output
        W_fc2 = weight_variable([size_fully_connected_layer, num_classes], "W_fc2")
        b_fc2 = bias_variable([num_classes], "b_fc2")
        l2_loss = 0.0
        l2_loss += tf.nn.l2_loss(W_fc2)
        l2_loss += tf.nn.l2_loss(b_fc2)

        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        
#        self.y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

#        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.train = tf.train.GradientDescentOptimizer(1e-4).minimize(self.cross_entropy)
        
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1)), tf.float32) + 1e-6 * l2_loss
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        
    def load_data(self, dir='/home/anshul/powerSpectrograms200/'):
        self.X_train = np.zeros((800, self.image_x, self.image_y, self.image_channels))
        self.Y_train = np.zeros((800,), dtype=int)
        self.X_test = np.zeros((200, self.image_x, self.image_y, self.image_channels))
        self.Y_test = np.zeros((200,), dtype=int)

        genres = {'blues':     0,
                  'classical': 1,
                  'country':   2,
                  'disco':     3,
                  'hiphop':    4,
                  'jazz':      5,
                  'metal':     6,
                  'pop':       7,
                  'reggae':    8,
                  'rock':      9}

        indexTrain = 0
        indexTest = 0
        for genre in genres.keys():
            for count in range(0, 100):
                path = dir + genre + '/' + genre + '.%0.5d' % count + '.au.png'
                if os.path.isfile(path):
                    if count < 80:
                        self.X_train[indexTrain] = plt.imread(path)
                        self.Y_train[indexTrain] = genres[genre]
                        indexTrain += 1
                    else:
                        self.X_test[indexTest] = plt.imread(path)
                        self.Y_test[indexTest] = genres[genre]
                        indexTest += 1
                        
        Y_train_onehot = np.zeros((self.Y_train.shape[0], self.num_classes))
        Y_train_onehot[np.arange(self.Y_train.shape[0]), self.Y_train] = 1
        
        Y_test_onehot = np.zeros((self.Y_test.shape[0], self.num_classes))
        Y_test_onehot[np.arange(self.Y_test.shape[0]), self.Y_test] = 1
        
        self.Y_train = Y_train_onehot
        self.Y_test = Y_test_onehot
        
        self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)
                        

cnn = CNN(image_x=200,
          image_y=200,
          image_channels=4,
          num_classes=10,
          num_filters_fist_layer=80,
          num_filters_second_layer=80,
          patch_size=5,
          size_fully_connected_layer=40)


# In[ ]:

cnn.load_data()


# In[ ]:

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)

    train_batches = 16
    test_batches = 10

    n_train_images = 800
    n_test_images = 200

    train_step_size = int(n_train_images / train_batches)
    test_step_size = int(n_test_images / test_batches)

    for i in range(0, n_train_images, train_step_size):
        print("Batch", i, "to", i + train_step_size - 1)
        X_train = cnn.X_train[i : i + train_step_size]
        Y_train = cnn.Y_train[i : i + train_step_size]
        
        feed_dict = {cnn.x : X_train, cnn.y_ : Y_train, cnn.keep_prob : 1.0}
        session.run(cnn.train, feed_dict)
        
        trainAccuracy = session.run(cnn.accuracy, feed_dict)
        print("Train Accuracy:", trainAccuracy)
        
        feed_dict = {cnn.x : cnn.X_test, cnn.y_ : cnn.Y_test, cnn.keep_prob : 1.0}
        testAccuracy = session.run(cnn.accuracy, feed_dict)
        print("Test Accuracy: ", testAccuracy)


# In[ ]:



