from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time, os
'''
Valid labels:
    [ 0  1  2  4  5  6  7  8 10 11 |no 17]
'''
tf.logging.set_verbosity(tf.logging.ERROR)
num_style_dim = 32
classes_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
num_classes = 11
img_size = [96, 96]
BATCH_SIZE = 15
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
    print(features)
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
    x = tf.reshape(x, img_size)
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
    print(x)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=512, activation=activation)
    # x=tf.layers.dropout(x,0.5)
    x = tf.layers.dense(inputs=x, units=256, activation=activation)
    # x=tf.layers.dropout(x,0.5)
    x = tf.layers.dense(inputs=x, units=num_style_dim, activation=activation)
    x = tf.nn.softmax(x, name='output')
    return x, phase_train


def decoder(features):
    activation = tf.nn.relu
    # masks /= num_classes
    # masks -= num_classes / 2
    x = tf.layers.dense(x, 256, activation=activation)
    x = tf.layers.dense(
        inputs=x,
        units=img_size[0] / 4 * img_size[1] / 4 * 3,
        activation=activation)
    x = tf.reshape([img_size[0], img_size[1], 3])
    x = tf.layers.conv2d_transpose(
        x,
        filters=6,
        kernel_size=[2, 2],
        strides=(2, 2),
        padding='valid',
    )
    x = tf.layers.conv2d_transpose(
        x,
        filters=6,
        kernel_size=[2, 2],
        strides=(2, 2),
        padding='valid',
    )
    return x


def discriminator(x):
    activation = tf.nn.relu
    x = tf.reshape(x, img_size)
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
    print(x)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=512, activation=activation)
    # x=tf.layers.dropout(x,0.5)
    x = tf.layers.dense(inputs=x, units=256, activation=activation)
    # x=tf.layers.dropout(x,0.5)
    x = tf.layers.dense(inputs=x, units=2, activation=activation)
    x = tf.nn.softmax(x, name='output')
    return x, phase_train


def main(unused_argv):
    plt.ion()
    image, mask = input_pipeline(
        ['./pascalvoc2012_96x.tfrecords'], batch_size=BATCH_SIZE)
    print(image)
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
    for i in classes_labels:
        mask_class = tf.cast(tf.equal(mask, i), tf.float32) * ((mask + 1) /
                                                               (i + 1))
        rec = tf.cond(
            tf.reducesum(mask_class) > 0,
            lambda: decoder(encoder(image * mask_class)),
            lambda: tf.zeros([BATCH_SIZE, img_size[0], img_size[1], 3]))
        reconstructions.append(rec)
    reconstructions=tf.stack(reconstructions)
    print(reconstructions)
    os.ex

    print('Y: {}'.format(y))

    #Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, y)
    # loss = tf.losses.absolute_difference(tf.one_hot(labels, depth=2), y)
    cross_entropy = tf.reduce_mean(loss)

    #Optimizer
    rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

    #Accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), labels)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    #Saver
    save_dir = './models/model'
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        t0 = time.clock()
        rt = 3e-6 * 8
        for i in range(60001):
            # Train
            data_feed = train_data_batch[i % (len(train_data_batch) - 1)]
            label_feed = train_data_label[i % (len(train_data_label) - 1)]
            _, l = sess.run(
                [train_step, cross_entropy],
                feed_dict={
                    input_data: data_feed,
                    labels: label_feed,
                    phase_train: True,
                    rate: rt
                })
            if (i % 60 == 0) and (i != 0):
                rt = 3e-6 * 4
                # Print the accuracy
                acc = 0.
                test_loss = 0
                for j in range(len(test_data_batch)):
                    _, loss_once, acc_once = sess.run(
                        [train_step, cross_entropy, accuracy],
                        feed_dict={
                            input_data:
                            test_data_batch[j % len(test_data_batch)],
                            labels: test_data_label[j % len(test_data_label)],
                            phase_train: False,
                            rate: rt
                        })
                    acc += acc_once
                    test_loss += loss_once
                print('%g, %g, %g, %g, %g' %
                      (i / 60, l, test_loss / len(test_data_batch),
                       acc / len(test_data_batch), (time.clock() - t0)))
                t0 = time.clock()
                if i / 60 >= 9 and input("Stop?(y/n)") == 'y': break
        builder.add_meta_graph_and_variables(
            sess, tf.saved_model.tag_constants.TRAINING)
        builder.save()
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
