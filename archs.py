"""
file containing different architectures for generators and discriminators
"""
import tensorflow as tf

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def discriminator_vanilla(input, im_dim, reuse=False):
    """
    discriminator used for mnist images of 28 x 28
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h0 = tf.layers.dense(input, im_dim*im_dim, tf.nn.relu, name='d0')
    h1 = tf.layers.dense(h0, 1,  name='d1')
    prob = tf.sigmoid(h1, 'd2')
    return prob
        
def generator_vanilla(z, z_dim, im_dim):
    """
    discriminator used for mnist images of 28 x 28
    """
    h0 = tf.layers.dense(z, z_dim, tf.nn.relu, name='g0')
    h1 = tf.layers.dense(h0, im_dim*im_dim, name='g1')
    h2 = tf.sigmoid(h1, name='g2')
    return h2


def generator_conv(z, is_training=None):
    """
    generetor for 28 x 28 grayscale images
    """
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x
    
def discriminator_conv(input, im_dim, reuse=None):
    """
    discriminator for 28 x 28 grayscale images
    """
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(input, shape=[-1, im_dim, im_dim, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x