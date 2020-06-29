import tensorflow as tf
import os
import scipy.misc
import cifar10_input



def inputs_origin(data_dir):
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    for f in filenames:
        print(f)
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file' + f)
    filenames_queue =tf.train.string_input_producer(filenames)
    read_input = cifar10_input.read_cifar10(filenames_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)
    print(reshaped_image)
    return reshaped_image

if __name__ == '__main__':
    with tf.Session() as sess:
        reshaped_image = inputs_origin('cifar-10-batches-py')
        threads = tf.train.start_queue_runners(sess=sess)
        print(threads)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('cifar-10-batches-py/raw/'):
            os.makedirs('cifar-10-batches-py/raw/')
        for i in range(30):
            image = sess.run(reshaped_image)
            scipy.misc.toimage(image).save('cifar-10-batches-py/raw/%d.jpg' %i)