#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

# inputsize = [1,224,224,3]
# netinput = tf.placeholder(tf.float32,inputsize)

# # Padding
# paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]])
# padout = tf.pad(netinput, paddings)

# # Graph 1
# # Convolution
# kernel_shape = [5,5,3,3]
# weight = tf.random_normal(shape=kernel_shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=tf.set_random_seed(1234))
# kernel = tf.get_variable('weight', kernel_shape, dtype=tf.float32)
# conv_out1 = tf.nn.atrous_conv2d(netinput, kernel, padding='VALID', rate=2)
# # conv_out1 = tf.nn.conv2d(padout, kernel, strides=[1,1,1,1], padding='VALID')
# netout1 = tf.nn.max_pool(conv_out1, [1,5,5,1], [1,2,2,1], padding='VALID')

# # Graph 2
# conv_out2 = tf.nn.conv2d(netinput, kernel, strides=[1,1,1,1], padding='SAME')
# netout2 = tf.nn.max_pool(conv_out2, [1,5,5,1], [1,2,2,1], padding='VALID')

# shape = [1,55,55,12]
# reshape_out = tf.reshape(netout, shape)

# split_out = tf.split(reshape_out, num_or_size_splits=3, axis=3)
kernel =  np.random.random((5,5,3,3)).astype(np.float32)
undef_input_op = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
undef_output_op = tf.nn.atrous_conv2d(undef_input_op, kernel, padding='VALID', rate=2)
netout2 = tf.nn.max_pool(undef_output_op, [1,5,5,1], [1,2,2,1], padding='VALID')

############# session #######################
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
sess = tf.Session()

# sess.run(init)
# saver.save(sess, "./sample_convnet.ckpt")

tf.train.write_graph(sess.graph, '.', 'sample_atrous_conv.pb', as_text=False)
sess.close()
