'''
This file contains the main function to build model and train model.

'''



from utils import *
from ops import *
from datetime import timedelta

import numpy as np
import tensorflow as tf
import time


BATCH_SIZE = 1
LAMBDA = 10
OUTPUT_PATH = 'results'



# define dataset class
Data = Dataset()

# define placeholder variables
input_A = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 3], name='input_A')
input_B = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 3], name='input_B')
fake_pool_A = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_pool_A")
fake_pool_B = tf.placeholder(tf.float32, [None, 256, 256, 3], name="fake_pool_B")
learning_rate_g = tf.placeholder(tf.float32, name='learning_rate_g')
learning_rate_d = tf.placeholder(tf.float32, name='learning_rate_d')
# if you use batch normalization, it is different during testing and training.
# if you use instance normalization, there is no difference.
is_train = tf.placeholder(tf.bool, name='is_train')


# real_A to fake_B
gen_B = generator(input_A, is_train=is_train, name='generator_AtoB')
dec_B_real = discriminator(input_B, is_train=is_train, name='discriminator_B')
dec_B_fake = discriminator(gen_B, is_train=is_train, reuse=True, name='discriminator_B')

# real_B to fake_A
gen_A = generator(input_B, is_train=is_train, name='generator_BtoA')
dec_A_real = discriminator(input_A, is_train=is_train, name='discriminator_A')
dec_A_fake = discriminator(gen_A, is_train=is_train, reuse=True, name='discriminator_A')

# fake_B to real_A
gen_A_from_fake_B = generator(gen_B, is_train=is_train, reuse=True, name='generator_BtoA')

# fake_A to real_B
gen_B_from_fake_A = generator(gen_A, is_train=is_train, reuse=True, name='generator_AtoB')

# discriminate fake_pool_A
dec_A_pool_fake = discriminator(fake_pool_A, is_train=is_train, reuse=True, name='discriminator_A')

# discriminate fake_pool_B
dec_B_pool_fake = discriminator(fake_pool_B, is_train=is_train, reuse=True, name='discriminator_B')



# add summary
temp1 = (input_A + 1)*127.5
temp2 = (gen_A + 1)*127.5
temp3 = (input_B + 1)*127.5
temp4 = (gen_B + 1)*127.5
temp5 = (gen_A_from_fake_B + 1)*127.5
temp6 = (gen_B_from_fake_A + 1)*127.5

tempA = tf.concat((temp1, temp4, temp5), axis=2)
tempB = tf.concat((temp3, temp2, temp6), axis=2)
tf.summary.image('results_A/toB/backA', tf.cast(tempA, tf.uint8), max_outputs=1)
tf.summary.image('results_B/toA/backB', tf.cast(tempB, tf.uint8), max_outputs=1)


# define cycle loss
cycle_loss_A = cycle_consistency_loss(gen_A_from_fake_B, input_A)
cycle_loss_B = cycle_consistency_loss(gen_B_from_fake_A, input_B)
cycle_loss = cycle_loss_A + cycle_loss_B

# define gan loss
loss_gen_A_1 = generator_loss(dec_A_fake, use_lsgan=False)
loss_gen_B_1 = generator_loss(dec_B_fake, use_lsgan=False)


# define generator loss
loss_gen_A = loss_gen_A_1 + LAMBDA * cycle_loss
loss_gen_B = loss_gen_B_1 + LAMBDA * cycle_loss

# define discriminator loss
loss_dec_A = discriminator_loss(dec_A_real, dec_A_pool_fake, use_lsgan=False)
loss_dec_B = discriminator_loss(dec_B_real, dec_B_pool_fake, use_lsgan=False)


# add summary
tf.summary.scalar('loss_dec_A', loss_dec_A)
tf.summary.scalar('loss_dec_B', loss_dec_B)
tf.summary.scalar('regular_gan_loss_A', loss_gen_A_1)
tf.summary.scalar('regular_gan_loss_B', loss_gen_B_1)
tf.summary.scalar('cycle_loss', cycle_loss)


# retrieve variables
var_g_A = [item for item in tf.trainable_variables() if item.name.startswith('generator_AtoB')]
var_d_A = [item for item in tf.trainable_variables() if item.name.startswith('discriminator_A')]
var_g_B = [item for item in tf.trainable_variables() if item.name.startswith('generator_BtoA')]
var_d_B = [item for item in tf.trainable_variables() if item.name.startswith('discriminator_B')]

# define optimizer function
op_gen_A = optimizer(loss_gen_A, var_g_A, lr=learning_rate_g, name='op_gen_A')
op_dec_A = optimizer(loss_dec_A, var_d_A, lr=learning_rate_d, name='op_dec_A')
op_gen_B = optimizer(loss_gen_B, var_g_B, lr=learning_rate_g, name='op_gen_B')
op_dec_B = optimizer(loss_dec_B, var_d_B, lr=learning_rate_d, name='op_dec_B')


# define session
session = tf.Session()
session.run(tf.global_variables_initializer())

# initialize saver to store model
saver = tf.train.Saver()

# add to tensorboard
writer = tf.summary.FileWriter('Tensorboard/apple2orange')
writer.add_graph(session.graph)
merge = tf.summary.merge_all()


t1 = time.time()
num_fake_inputs_A = 0
num_fake_inputs_B = 0
temp_fake_pool_A = None
temp_fake_pool_B = None

# begin training
for i in range(70001):

    # decrease learning rate after 50 epoch
    if i < 5e4:
        lr_g = 0.0002
        lr_d = 0.0002
    else:
        lr_g = 0.0002 - 0.0002*(i - 5e4)/5e4
        lr_d = 0.0002 - 0.0002*(i - 5e4)/5e4


    d_A, d_B = Data.next_batch()

    feed_dict={input_A:d_A, input_B:d_B, is_train:True}
    fake_A_temp, fake_B_temp = session.run([gen_A, gen_B], feed_dict=feed_dict)
    temp_fake_pool_A = fake_image_pool(num_fake_inputs_A, fake_A_temp, temp_fake_pool_A)
    temp_fake_pool_B = fake_image_pool(num_fake_inputs_B, fake_B_temp, temp_fake_pool_B)
    num_fake_inputs_A += 1
    num_fake_inputs_B += 1


    # write to tensorboard
    if i%10 == 0:
        feed_dict = {input_A:d_A, input_B:d_B, is_train:True, learning_rate_g:lr_g, learning_rate_d:lr_d,
                    fake_pool_A:temp_fake_pool_A, fake_pool_B:temp_fake_pool_B}

        im, result = session.run([gen_A_from_fake_B, merge], feed_dict=feed_dict)
        writer.add_summary(result, i)


    # Optimizing the G_B network
    _, temp_B = session.run([op_gen_B, gen_B], feed_dict={input_A:d_A, input_B:d_B, is_train:True,
                            learning_rate_g:lr_g, learning_rate_d:lr_d})

    temp_fake_pool_B = fake_image_pool(num_fake_inputs_B, temp_B, temp_fake_pool_B)
    num_fake_inputs_B += 1


    # Optimizing the D_B network
    session.run([op_dec_B],feed_dict={input_B:d_B, is_train:True, learning_rate_g:lr_g, learning_rate_d:lr_d,
                                    fake_pool_B:temp_fake_pool_B})

    # Optimizing the G_A network
    _, temp_A = session.run([op_gen_A, gen_A], feed_dict={input_A:d_A, input_B:d_B, is_train:True,
                            learning_rate_g:lr_g, learning_rate_d:lr_d})

    temp_fake_pool_A = fake_image_pool(num_fake_inputs_A, temp_A, temp_fake_pool_A)
    num_fake_inputs_A += 1


    # Optimizing the D_A network
    session.run([op_dec_A],feed_dict={input_A:d_A, is_train:True, learning_rate_g:lr_g, learning_rate_d:lr_d,
                                    fake_pool_A:temp_fake_pool_A})


    # save results of testing dataset
    if i%100 == 0:
        d_A, d_B = Data.get_random_test_batch(BATCH_SIZE)
        feed_dict = {input_A:d_A, input_B:d_B, is_train:True}

        t2 = time.time()
        time_dif = t2 - t1
        print('Iteration {}'.format(i))
        print('Time usage: {}...'.format(timedelta(seconds=int(time_dif))))
        print()

        gen_A_image, gen_B_image = session.run([gen_A, gen_B], feed_dict=feed_dict)
        images = np.concatenate((d_A, gen_B_image, d_B, gen_A_image), axis=0)
        deprocess_and_save_result(images, i, output_path=OUTPUT_PATH)

    # save model every 10000 iteration
    if i%10000 == 0 and i > 1:
        path = os.path.join(os.getcwd(), 'model/MODEL.ckpt')
        saver.save(session, path, global_step=i)
        print('temporal model saved.')




print('finished.')
