import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 64
n_noise = 64
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
z = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])
    
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def discriminator(img_in, reuse=None):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x

def generator(z, is_training=is_training):
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
    

g = generator(z, is_training)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


'''
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)


loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))
'''

loss_d = - tf.reduce_mean(tf.log(d_real) + tf.log(1.0 - d_fake))
loss_g = - tf.reduce_mean(tf.log(d_fake))
    
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d, var_list=vars_d)
    g_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g, var_list=vars_g)
    
    
num_steps = 100000
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    for i in range(num_steps):
        
        if i % 100 == 0:
            save_path = saver.save(sess, "tmp/model.ckpt")

        
        if i % 100 == 0:
            fake_data = sess.run(g, {
                z: sample_Z(16, n_noise),
                is_training: False
            })
            fig = plot(fake_data)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)
        
        
        train_data, _ = mnist.train.next_batch(batch_size)
        train_data = np.reshape(train_data, (-1, 28, 28))

        #update discriminator
        loss_disc, _ = sess.run([loss_d, d_optimizer], {
            X_in: train_data,
            z: sample_Z(batch_size, n_noise),
            is_training:True
        })
        
        #update generator
        loss_gen, _ = sess.run([loss_g, g_optimizer], {
            z: sample_Z(batch_size, n_noise),
            is_training:True
        })
    
        #update generator
        loss_gen, _ = sess.run([loss_g, g_optimizer], {
            z: sample_Z(batch_size, n_noise),
            is_training:True
        })
        

        if i % 100 == 0:
            print(loss_disc, loss_gen)
    
    
