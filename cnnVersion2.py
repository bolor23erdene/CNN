import os
import numpy as np
import math

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print(X_test.shape,Y_test.shape)

img_size = 28
img_chan = 1
n_classes = 10

X = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
Y_ = tf.placeholder(tf.float32, (None, n_classes), name='y')

def model(x, logits=False):

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=24, kernel_size=[6, 6], strides=(1,1), padding='same', activation=tf.nn.relu)
        #z = tf.layers.max_pooling2d(z, pool_size=[1, 1], strides=1)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=48, kernel_size=[5, 5], strides=(2,2), padding='same', activation=tf.nn.relu)
        #z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv2'):
        z = tf.layers.conv2d(x, filters=64, kernel_size=[4, 4], strides=(2,2), padding='same', activation=tf.nn.relu)
        #z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flat'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('dense'):
        z = tf.layers.dense(z, units=200, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits: #if logits is True
        return y, logits_
    return y 



y_pred, logits = model(X, logits=True)

    ### accuracy ### 
    ### 0.98 ### 0.01 ###

count = tf.equal(tf.argmax(Y_, axis=1), tf.argmax(y_pred, axis=1))
acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc') 

    ### loss ###
    ### 0.062 ### 12.123 ###
xent = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=logits)
loss = tf.reduce_mean(xent, name='loss')

    ### optimize ###
    ### update the weights ###
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)

#################################################################################

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

n_epochs = 16
batch_size = 128
n_batch = math.floor(len(X_train)/batch_size)

for i in range(n_epochs):
    for batch in range(n_batch):

        start = batch_size * batch
        end = start + batch_size

        batch_X = X_train[start:end]
        batch_Y = Y_train[start:end]

        sess.run(train_op, feed_dict = {X: batch_X, Y_: batch_Y})

        if batch % 200 == 0:
            print('Epoch number: ' + str(i+1) + 
                  ' Batch Number: ' + str(batch) + 
                  ' Accuracy: ' + str(sess.run(acc, feed_dict = {X: X_test, Y_: Y_test})) + 
                  ' Loss: ' + str(sess.run(loss, feed_dict = {X: batch_X, Y_: batch_Y})) )

################################################################################

sess.close()
