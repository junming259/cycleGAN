'''
This file contains all operation function for later building model and training.
The convention of function's name is based on the original paper, which you can
see:
                https://arxiv.org/pdf/1703.10593.pdf

Network structures are explained in the appendix of the paper. Thanks for
vanhuyz. I use his code to debug my code.
            https://github.com/vanhuyz/CycleGAN-TensorFlow

Note:
In the original paper, the author uses regular batch normalization, in practice,
it is recommended to use instance normalization.


'''





import tensorflow as tf
import numpy as np
import os

from skimage.io import imsave
from random import shuffle



# name of folder which will be used to save reuslt images
FILTER_SIZE = 4
POOL_SIZE = 50



def batch_norm(inputs, is_train, use_instance=True):
    """
    Normalization.
    : param inputs: input layer for normalization
    : param is_train: it is a placeholder variable for batch normalization.
    : use_instance: bool, True to use instance normalization, otherwise regular
                    batch normalization. It is recommended in this certain
                    problem to use instance normalization.
    """
    if use_instance:

        with tf.variable_scope("instance_norm"):
            depth = inputs.get_shape()[3]
            scale = tf.get_variable("scale", shape=[depth],
                                    initializer=tf.random_normal_initializer(1.0, 0.02))

            offset = tf.get_variable("offset", shape=[depth],
                                    initializer=tf.zeros_initializer())

            mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (inputs-mean)*inv

            return scale*normalized + offset

    else:

        _batch_norm(inputs=inputs, is_train=is_train)



def _batch_norm(inputs, is_train, decay=0.9):
    '''
    batch normalization. I modify code from https://github.com/fzliu/tf-dcgan/blob/master/models.py
    I came cross an annoying bugs when running original code, which indicates
    that exponential moving average cannot be under a reuse=True scope. I avoid
    this problem by removing tf.train.ExponentialMovingAverage(), and implement
    mean and variance moving by hand.
    : param input: inputs layer
    : param is_train: an indicator which indicates whether it is in training mode
    : param decay: exponential parameter
    : return: normalized batch layer
    '''

    inputs = tf.identity(inputs)
    bn_shape = inputs.get_shape()[-1]
    shift = tf.get_variable("beta", shape=bn_shape,
                            initializer=tf.zeros_initializer())
    scale = tf.get_variable("gamma", shape=bn_shape,
                            initializer=tf.random_normal_initializer(1.0, 0.02))

    pop_mean = tf.get_variable("pop_mean", shape=bn_shape,
                            initializer=tf.zeros_initializer(),
                            trainable=False)

    pop_var = tf.get_variable("pop_var", shape=bn_shape,
                            initializer=tf.random_normal_initializer(1.0, 0.02),
                            trainable=False)

    def train_op():
        # training step
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    def test_op():
        # testing step
        return pop_mean, pop_var

    (mean, var) = tf.cond(is_train, train_op, test_op)

    return tf.nn.batch_normalization(inputs, mean, var, shift, scale, 1e-5)



def _leak_relu(inputs, alpha=0.2):
    '''
    leak relu activation function
    '''
    with tf.variable_scope('leak_relu'):
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)

    return f1 * inputs + f2 * tf.abs(inputs)



def leak_relu(inputs, alpha=0.2):
    '''
    leak relu activation function
    '''
    return tf.maximum(alpha*inputs, inputs)



def c7s1_k(inputs, k, stride=1, use_relu=True, with_bn=True, is_train=None, name='c7s1'):
    '''
    : param inputs: input layer
    : param k: size of output channels
    : param stride: size of stride
    : param use_relu: True to use relu activation function, otherwise not
    : param with_bn: decide whether to use normalization or not
    : param is_train: input variable for regular batch normalization
    '''

    with tf.variable_scope(name) as scope:
        kernel_shape = [7, 7, inputs.get_shape()[-1], k]
        weights = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        # use reflection padding
        padded = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
        outs = tf.nn.conv2d(padded, weights, strides=[1,stride,stride,1], padding='VALID')

        tf.summary.histogram('weights', weights)

        # post batch normalization
        if with_bn:
            outs = batch_norm(outs, is_train)

        # activation function
        if use_relu:
            outs = tf.nn.relu(outs)

    return outs



def dk(inputs, k, stride=2, use_relu=True, with_bn=True, is_train=None, name='dk'):
    '''
    : param inputs: input layer
    : param k: size of output channels
    : param stride: size of stride
    : param use_relu: True to use relu activation function, otherwise not
    : param with_bn: decide whether to use normalization or not
    : param is_train: input variable for regular batch normalization
    '''
    with tf.variable_scope(name) as scope:
        kernel_shape = [3, 3, inputs.get_shape()[3], k]
        weights = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        # use reflection padding
        # padded = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        # outs = tf.nn.conv2d(padded, weights, strides=[1,stride,stride,1], padding='VALID')
        outs = tf.nn.conv2d(inputs, weights, strides=[1,stride,stride,1], padding='SAME')


        tf.summary.histogram('weights', weights)

        # post batch normalization
        if with_bn:
            outs = batch_norm(outs, is_train)

        # activation function
        if use_relu:
            outs = tf.nn.relu(outs)

    return outs



def Rk(inputs, k, with_bn=True, is_train=None, name='Rk'):
    '''
    Residual block.

    : param inputs: input layer
    : param k: size of output channels
    : param with_bn: decide whether to use normalization or not
    : param is_train: input variable for regular batch normalization
    '''
    with tf.variable_scope(name):

        with tf.variable_scope('layer1'):
            kernel_shape = [3, 3, inputs.get_shape()[3], k]
            weights1 = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
            padded1 = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
            outs1 = tf.nn.conv2d(padded1, weights1, strides=[1,1,1,1], padding='VALID')

            tf.summary.histogram('weights', weights1)

            if with_bn:
                outs1 = batch_norm(outs1, is_train)
            relu1 = tf.nn.relu(outs1)


        with tf.variable_scope('layer2'):
            kernel_shape = [3, 3, relu1.get_shape()[3], k]
            weights2 = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
            padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
            outs2 = tf.nn.conv2d(padded2, weights2, strides=[1,1,1,1], padding='VALID')

            tf.summary.histogram('weights', weights2)


            if with_bn:
                outs2 = batch_norm(outs2, is_train)

        output = inputs + outs2
        # notice a little different from original residual block, with no activation
        # in the last layer
        # outsput = tf.nn.relu(output)

    return output



def uk(inputs, k, stride=2, use_relu=True, with_bn=True, is_train=None, name='uk'):
    '''
    : param inputs: input layer
    : param k: size of output channels
    : param stride: size of stride
    : param use_relu: True to use relu activation function, otherwise not
    : param with_bn: decide whether to use normalization or not
    : param is_train: input variable for regular batch normalization
    '''

    with tf.variable_scope(name) as scope:
        inputs_shape = inputs.get_shape().as_list()
        kernel_shape = [3, 3, k, inputs_shape[3]]
        weights = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_shape = [inputs_shape[0], inputs_shape[1]*2, inputs_shape[2]*2, k]

        # deconvolution
        outs = tf.nn.conv2d_transpose(inputs, weights, output_shape,
                                        strides=[1,stride,stride,1], padding='SAME')

        tf.summary.histogram('weights', weights)

        # post batch normalization
        if with_bn:
            outs = batch_norm(outs, is_train)

        # activation function
        if use_relu:
            outs = tf.nn.relu(outs)

    return outs



def Ck(inputs, k, stride=2, use_lrelu=False, with_bn=False, is_train=None, name='conv'):
    '''
    : param inputs: input layer
    : param k: size of output channels
    : param stride: size of stride
    : param use_lrelu: True to use leak_relu activation function, otherwise not
    : param with_bn: decide whether to use normalization or not
    : param is_train: input variable for regular batch normalization
    '''

    with tf.variable_scope(name) as scope:
        kernel_shape = [4, 4, inputs.get_shape()[3], k]
        weights = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        outs = tf.nn.conv2d(inputs, weights, strides=[1,stride,stride,1], padding='SAME')

        tf.summary.histogram('weights', weights)

        # post batch normalization
        if with_bn:
            outs = batch_norm(outs, is_train)

        # activation function
        if use_lrelu:
            outs = leak_relu(outs)

        return outs



def regular_conv(inputs, k, name='regular_conv'):
    '''
    Regular convolution layer.

    : param inputs: input layer
    : param k: size of output channels
    '''

    with tf.variable_scope(name) as scope:
        kernel_shape = [4,4,inputs.get_shape()[3], k]
        weights = tf.get_variable('weights', shape=kernel_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable("biases", shape=[k],
                                    initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        outs = conv + biases

    return outs



def discriminator(inputs, is_train, reuse=False, name='discriminator'):
    '''
    Define structure of discriminator. C64-C128-C256-C512

    '''

    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables()

        # 128x128x64
        conv1 = Ck(inputs, 64, use_lrelu=True, with_bn=False,
                    is_train=is_train, name='d_conv1')

        # 64x64
        conv2 = Ck(conv1, 128, use_lrelu=True, with_bn=True,
                    is_train=is_train, name='d_conv2')

        # 32x32
        conv3 = Ck(conv2, 256, use_lrelu=True, with_bn=True,
                    is_train=is_train, name='d_conv3')

        # 16x16x512
        conv4 = Ck(conv3, 512, use_lrelu=True, with_bn=True,
                    is_train=is_train, name='d_conv4')

        # 16x16x1
        last_layer = regular_conv(conv4, 1, name='last_layer')

        # convert to probability
        # outs = tf.nn.sigmoid(last_layer)
        # note: if use_lsgan, don't use sigmoid()
        outs = last_layer

    return outs



def generator(inputs, is_train, reuse=False, name='generator'):
    '''
    Define structure of generator.
    c7s1-32,d64,d128,R128,R128,R128,
    R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
    '''

    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables()

        # 256x256
        c7s1_32 = c7s1_k(inputs, 32, use_relu=True, with_bn=False,
                            is_train=is_train, name='c7s1_32')

        # 128x128
        d64 = dk(c7s1_32, 64, use_relu=True, with_bn=True, is_train=is_train, name='d64')

        # 64x64
        d128 = dk(d64, 128, use_relu=True, with_bn=True, is_train=is_train, name='d128')

        # 64x64
        r1 = Rk(d128, 128, with_bn=True, is_train=is_train, name='r1')

        # 64x64
        r2 = Rk(r1, 128, with_bn=True, is_train=is_train, name='r2')

        # 64x64
        r3 = Rk(r2, 128, with_bn=True, is_train=is_train, name='r3')

        # 64x64
        r4 = Rk(r3, 128, with_bn=True, is_train=is_train, name='r4')

        # 64x64
        r5 = Rk(r4, 128, with_bn=True, is_train=is_train, name='r5')

        # 64x64
        r6 = Rk(r5, 128, with_bn=True, is_train=is_train, name='r6')

        # 64x64
        r7 = Rk(r6, 128, with_bn=True, is_train=is_train, name='r7')

        # 64x64
        r8 = Rk(r7, 128, with_bn=True, is_train=is_train, name='r8')

        # 64x64
        r9 = Rk(r8, 128, with_bn=True, is_train=is_train, name='r9')

        # 128x128
        u64 = uk(r9, 64, use_relu=True, with_bn=True, is_train=is_train, name='u64')

        # 256x256
        u32 = uk(u64, 32, use_relu=True, with_bn=True, is_train=is_train, name='u32')

        # 256x256x3, note: not normalization and activation in the last layer
        c7s1_3 = c7s1_k(u32, 3, use_relu=False, with_bn=False,
                            is_train=is_train, name='c7s1_3')

        # tanh
        outs = tf.nn.tanh(c7s1_3)

    return outs



def discriminator_loss(d_real, d_fake, use_lsgan=True):
    '''
    : param d_real: discriminator's results on real images
    : param d_fake: discriminator's results on fake images
    : param use_lsgan: True to use the loss introduced in original paper,
                    otherwise use regular loss for GAN
    '''

    if use_lsgan:
        # use mean squared error
        error_real = tf.reduce_mean(tf.squared_difference(d_real, 0.9))
        error_fake = tf.reduce_mean(tf.square(d_fake))
        loss = (error_real + error_fake)
    else:
        error_real = -tf.reduce_mean(tf.log(d_real + 1e-12))
        error_fake = -tf.reduce_mean(tf.log(1 - d_fake + 1e-12))
        loss = error_real + error_fake

    return loss



def generator_loss(d_fake, use_lsgan=True):
    '''
    : param d_fake: discriminator's results on fake images
    : param use_lsgan: True to use the loss introduced in original paper,
                    otherwise use regular loss for GAN

    '''

    if use_lsgan:
        # use mean squared error
        loss = tf.reduce_mean(tf.squared_difference(d_fake, 0.9))
    else:
        # heuristic, non-saturating loss
        loss = -tf.reduce_mean(tf.log(d_fake + 1e-12))

    return loss



def cycle_consistency_loss(image, origin):
    '''
    cycle consistency loss (L1 norm)
    '''

    loss = tf.reduce_mean(tf.abs(image - origin))

    return loss



def optimizer(loss, variables, lr=0.0002, beta=0.5, name='optimizer'):
    '''
    Define optimizer.

    : param loss: loss which needs to be optimized
    : param vairbales: variables which need to be optimized
    '''

    op = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta, name=name).minimize(loss, var_list=variables)

    return op



def fake_image_pool(num_fakes, fake, fake_pool):
    '''
    This function saves the generated image to corresponding pool of images.
    In starting, it keeps on feeding the pool till it is full and then replace
    the last one with new one.
    '''

    if num_fakes == 0:
        return fake

    if(num_fakes < POOL_SIZE):
        fake_pool = np.concatenate((fake_pool, fake), axis=0)
        return fake_pool

    else :
        fake_pool = np.concatenate((fake_pool[1:], fake), axis=0)
        return fake_pool



def deprocess_and_save_result(batch_res, epoch, output_path, grid_shape=(2, 2), grid_pad=5):
    '''
    create an output grid to hold the images and save it.
    '''
    if batch_res.shape[0] == 1:
        img = (batch_res + 1) * 127.5
        img = img.astype(np.uint8)
        image = img[0]
        fname = "iteration{0}_result.png".format(epoch)
        imsave(os.path.join(output_path, fname), image)
        return

    (img_h, img_w) = batch_res.shape[1:3]
    grid_h = img_h * grid_shape[0] + grid_pad * (grid_shape[0] - 1)
    grid_w = img_w * grid_shape[1] + grid_pad * (grid_shape[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # loop through all generator outputs
    for i, res in enumerate(batch_res):
        if i >= grid_shape[0] * grid_shape[1]:
            break

        # deprocessing (tanh)
        img = (res + 1) * 127.5
        img = img.astype(np.uint8)

        # add the image to the image grid
        row = (i // grid_shape[0]) * (img_h + grid_pad)
        col = (i % grid_shape[1]) * (img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w, :] = img


    # save the output image
    fname = "iteration{0}_result.jpg".format(epoch)
    imsave(os.path.join(output_path, fname), img_grid)
