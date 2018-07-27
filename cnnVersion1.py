import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
factor = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[factor:]
X_train = X_train[:factor]
y_valid = y_train[factor:]
y_train = y_train[:factor]

def model(x, logits=False):

    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flat'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('dense'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits: #if logits is True
        return y, logits_
    return y 


class Environment():
	pass

env = Environment()

env.x = tf.placeholder(tf.float32, (None, 28, 28, 1))
env.y = tf.placeholder(tf.float32, (None, 10)) 
#env.training = 

env.ybar, logits = model(env.x, logits=True)

with tf.variable_scope('acc'):
	count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
	env.acc = tf.reduce_mean(tf.cast(count, tf.float32))

with tf.variable_scope('loss'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=logits)
	env.loss = tf.reduce_mean(cross_entropy)

with tf.variable_scope('train'):
	env.train_op = tf.train.AdamOptimizer(0.005).minimize(env.loss)

def training(sess, env, X_data, Y_data, X_valid = None, y_valid = None, batch=128, shuffle=True, epochs=8):
	Xshape = X_data.shape
	n_data = Xshape[0]
	n_batches = int(n_data/batch)

	print(X_data.shape)


	for ep in range(epochs):
		print('epoch number: ', ep)
		if shuffle:
			ind = np.arange(n_data)
			np.random.shuffle(ind)
			X_data = X_data[ind]
			Y_data = Y_data[ind]

		for i in range(n_batches):
			start = i*batch 
			end = min(start+batch, n_data)
			#batch_X = X_data[start:end]
			#batch_Y = Y_data[start:end]

			sess.run(env.train_op, feed_dict={env.x: X_data[start:end], env.y: Y_data[start:end]})

		evaluate(sess, env, X_valid, y_valid)
		#if X_valid is not None:
        # 	evaluate(sess, env, X_valid, y_valid)



def evaluate(sess, env, X_test, Y_test, batch=128):
	n_data = X_test.shape[0]
	n_batches = int(n_data/batch)

	totalAcc = 0
	totalLoss = 0


	for i in range(n_batches):
		#print('batch ', i)
		start = i*batch 
		end = min(start+batch, n_data)
		batch_X = X_test[start:end]
		batch_Y = Y_test[start:end]

		batch_loss, batch_acc = sess.run([env.loss, env.acc], feed_dict={env.x: batch_X, env.y: batch_Y})

		totalAcc = totalAcc + batch_acc*(end-start)
		totalLoss = totalLoss + batch_loss*(end-start)
		

	totalAcc = totalAcc/n_data
	totalLoss = totalLoss/n_data
	print('acc: {0:.3f} loss: {1:.3f}'.format(totalAcc, totalLoss))
	return totalAcc, totalLoss

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training(sess, env, X_train, y_train, X_valid, y_valid, shuffle=True, batch=128)
evaluate(sess, env, X_test, y_test)
	


