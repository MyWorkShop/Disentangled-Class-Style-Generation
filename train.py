from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time, os
import colored_traceback
colored_traceback.add_hook()
'''
Valid labels:
    [ 0  1  2  4  5  6  7  8 10 11 |no 17]
'''
tf.logging.set_verbosity(tf.logging.ERROR)
num_style_dim = 32
classes_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
num_classes = 11
img_size = [96, 96]
BATCH_SIZE = 3
phase_train = tf.placeholder(tf.bool)

global fig
fig = plt.figure()


def imshow(img, block=True, rgb=False):
    global fig
    plt.figure(fig.number)
    if not rgb:
        plt.imshow(img, interpolation='nearest')
    else:
        plt.imshow(img)
    plt.show(block=block)


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        })
    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.float32)
    segmentation = tf.decode_raw(features['mask_raw'], tf.float32)

    image.set_shape([img_size[0] * img_size[1] * 3])
    segmentation.set_shape([img_size[0] * img_size[1] * 1])

    image = tf.reshape(image, [img_size[0], img_size[1], 3])
    segmentation = tf.reshape(segmentation, [img_size[0], img_size[1], 1])

    rgb = tf.cast(image, tf.float32)
    rgb = rgb / 255.
    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.int64)
    return rgb, mask


def input_pipeline(filenames, batch_size=15, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=False)

    image, label = read_and_decode(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    images_batch, labels_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=32,
        allow_smaller_final_batch=True)
    return images_batch, labels_batch


def encoder(x):
    activation = tf.nn.relu
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1, img_size[0], img_size[1], 3])
        x = tf.layers.conv2d(
            x,
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='valid',
        )
        x = tf.layers.average_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
        x = tf.layers.conv2d(
            x,
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='valid',
        )
        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs=x, units=256, activation=activation)
        # x=tf.layers.dropout(x,0.5)
        x = tf.layers.dense(inputs=x, units=256, activation=activation)
        # x=tf.layers.dropout(x,0.5)
        x = tf.layers.dense(
            inputs=x, units=num_style_dim, activation=activation)
        x = tf.nn.softmax(x, name='output')
        print('Encoder out: {}'.format(x))
        return x


def decoder(x):
    print('Decoder in: {}'.format(x))
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
        activation = tf.nn.relu
        x = tf.layers.dense(x, 256, activation=activation)
        x = tf.layers.dense(
            inputs=x,
            units=img_size[0] / 4 * img_size[1] / 4 * 3,
            activation=activation)
        x = tf.reshape(x, [-1, int(img_size[0] / 4), int(img_size[1] / 4), 3])
        x = tf.layers.conv2d_transpose(
            x,
            filters=6,
            kernel_size=[2, 2],
            strides=(2, 2),
            padding='valid',
        )
        x = tf.layers.conv2d_transpose(
            x,
            filters=3,
            kernel_size=[2, 2],
            strides=(2, 2),
            padding='valid',
        )
        print('Decoder out: {}'.format(x))
        return x


def discriminator(x):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        activation = tf.nn.relu
        x = tf.reshape(x, [-1, img_size[0], img_size[1], 3])
        x = tf.layers.conv2d(
            x,
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='valid',
        )
        x = tf.layers.average_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
        x = tf.layers.conv2d(
            x,
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='valid',
        )
        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs=x, units=256, activation=activation)
        # x=tf.layers.dropout(x,0.5)
        x = tf.layers.dense(inputs=x, units=256, activation=activation)
        # x=tf.layers.dropout(x,0.5)
        x = tf.layers.dense(inputs=x, units=2, activation=activation)
        x = tf.nn.softmax(x, name='output')
        return x


def main(unused_argv):
    plt.ion()
    image, mask = input_pipeline(
        ['./pascalvoc2012_96x.tfrecords'], batch_size=BATCH_SIZE)
    if False:
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            labels = [0]
            for i in range(500):
                im, ms = sess.run([image, mask])
                labels = np.unique(np.concatenate([labels, ms.flatten()]))
            coord.request_stop()
            coord.join(threads)
            print(labels)
        return

    # features = []
    reconstructions = []
    mask = tf.cast(mask, tf.float32)
    for i in classes_labels:
        mask_class = tf.cast(tf.equal(mask, i), tf.float32) * ((mask + 1) /
                                                               (i + 1))
        rec = tf.cond(
            tf.reduce_sum(mask_class) > 0,
            lambda: decoder(encoder(image * mask_class)),
            lambda: tf.zeros([BATCH_SIZE, img_size[0], img_size[1], 3]))
        reconstructions.append(rec)
    # TODO: Reduce max?
    reconstructions = tf.reduce_mean(tf.stack(reconstructions), axis=0)
    print(reconstructions)
    discriminator_out = discriminator(
        tf.concat([image, reconstructions], axis=0))
    discrimination_truth = tf.one_hot(
        tf.concat(
            [
                tf.ones(BATCH_SIZE, dtype=tf.int64),
                tf.zeros(BATCH_SIZE, dtype=tf.int64)
            ],
            axis=0),
        depth=2)
    #Loss
    rec_loss = tf.losses.mean_squared_error(image, reconstructions)
    generation_loss = -tf.losses.sigmoid_cross_entropy(discriminator_out,
                                                       discrimination_truth)

    disc_loss = tf.losses.sigmoid_cross_entropy(discriminator_out,
                                                discrimination_truth)
    loss_1 = tf.reduce_mean(rec_loss) + tf.reduce_mean(generation_loss)
    loss_disc = tf.reduce_mean(disc_loss)

    tensor_names = [
        t.name for op in tf.get_default_graph().get_operations()
        for t in op.values()
    ]
    # for n in tensor_names:
    # print(n)

    #Optimizer
    rate = tf.placeholder(tf.float32)
    step_1_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder') + tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Encoder')
    train_step_1 = tf.train.AdamOptimizer(rate).minimize(
        loss_1, var_list=step_1_vars)
    step_disc_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')
    train_step_disc = tf.train.AdamOptimizer(rate).minimize(
        loss_disc, var_list=step_disc_vars)

    def train_1(rt, sess):
        _, l = sess.run(
            [train_step_1, loss_1],
            feed_dict={
                rate: rt,
                phase_train: True
            },
        )
        return l

    def train_disc(rt, sess):
        _, l = sess.run(
            [train_step_disc, loss_disc],
            feed_dict={
                rate: rt,
                phase_train: True
            },
        )
        return l

    #Saver
    save_dir = './models/model'
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 5e-6
        l1 = 0
        ld = 0
        for i in range(60001):
            # Train
            if (i % 10 == 0):
                ld = train_disc(rt, sess)
            l1 = train_1(rt, sess)
            if (i % 300 == 0):
                print('Epoch: {}, Loss1: {}, LossD: {}, Time: {}'.format(
                    i, l1, ld,
                    time.clock() - t0))
                t0 = time.clock()
        builder.add_meta_graph_and_variables(
            sess, tf.saved_model.tag_constants.TRAINING)
        builder.save()
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
