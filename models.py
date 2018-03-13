import tensorflow as tf
import matplotlib.pyplot as plt 
from gan_utils import sample_z, plot_im, xavier_init
import archs
import numpy as np

class MNIST_GAN():
    def __init__(self, model_name, batch_size, dataset, num_iter, z_dim,
                 im_dim, steps):
        self.dataset = dataset
        self.num_iter = num_iter
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
      

    def train(self):
        tf.reset_default_graph() #clear graph
    
        with tf.variable_scope('generator'):
            z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
            gen_data = archs.generator_vanilla(z, self.z_dim, self.im_dim)
            #gen_data = self.generator(z)
       
        
        with tf.variable_scope('discriminator'):
            x = tf.placeholder(\
              tf.float32, shape=[None, self.im_dim*self.im_dim], name='inputs')
            d_real = archs.discriminator_vanilla(x, self.im_dim,  reuse=False)
            d_fake = archs.discriminator_vanilla(gen_data, self.im_dim,  reuse=True)
    

        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator/')]
        g_params = [v for v in vars if v.name.startswith('generator/')]
        
        loss_d = - tf.reduce_mean(tf.log(d_real) + tf.log(1.0 - d_fake))
        loss_g = - tf.reduce_mean(tf.log(d_fake))
            
        d_optimizer = tf.train.AdamOptimizer().minimize(loss_d, var_list=d_params)
        g_optimizer = tf.train.AdamOptimizer().minimize(loss_g, var_list=g_params)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, "tmp/" + self.model_name + ".ckpt")
            except:
                print("Failed to restore")
                init = tf.global_variables_initializer()
                sess.run(init)
            
            for i in range(self.num_iter):
                
                train_data, _ = self.dataset.next()

                #update discriminator
                loss_disc, _ = sess.run([loss_d, d_optimizer], {
                    x: train_data,
                    z: sample_z(self.batch_size, self.z_dim)
                })
        
                #update generator
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                    z: sample_z(self.batch_size, self.z_dim)
                })
    
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                    z: sample_z(self.batch_size, self.z_dim)
                })
    
                if i % self.steps["save"] == 0:
                    saver.save(sess, "tmp/" + self.model_name + ".ckpt")
        
                
                if i % self.steps["sample"] == 0:
                    fake_data = sess.run(gen_data, {
                            z: sample_z(16, self.z_dim)
                    })
                    fig = plot_im(fake_data)
                    plt.savefig('out/' + self.model_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print(loss_disc, loss_gen)
                    
class MNIST_DCGAN():
    def __init__(self, model_name, batch_size, dataset, num_iter, z_dim,
                 im_dim, steps):
        self.dataset = dataset
        self.num_iter = num_iter
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
        
    def train(self):
        tf.reset_default_graph()
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        
        with tf.variable_scope('generator'):
            z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])    
            gen_data = archs.generator_conv(z, is_training)

        with tf.variable_scope('generator'):
            x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='x')        
       
        d_real = archs.discriminator_conv(x, self.im_dim, reuse=False)
        d_fake = archs.discriminator_conv(gen_data, self.im_dim, reuse=True)
        
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator/')]
        g_params = [v for v in vars if v.name.startswith('generator/')]
           
        loss_d = - tf.reduce_mean(tf.log(d_real) + tf.log(1.0 - d_fake))
        loss_g = - tf.reduce_mean(tf.log(d_fake))
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=0.00015).minimize(loss_d, var_list=d_params)
            g_optimizer = tf.train.RMSPropOptimizer(\
                learning_rate=0.00015).minimize(loss_g, var_list=g_params)
            
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, "tmp/" + self.model_name + ".ckpt")
            except:
                print("Failed to restore")
                init = tf.global_variables_initializer()
                sess.run(init)
            
            for i in range(self.num_iter):
                
                train_data, _ = self.dataset.next()
                train_data = np.reshape(train_data, (-1, 28, 28))

                #update discriminator
                loss_disc, _ = sess.run([loss_d, d_optimizer], {
                    x: train_data,
                    z: sample_z(self.batch_size, self.z_dim),
                    is_training:True
                })
        
                #update generator
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                        z: sample_z(self.batch_size, self.z_dim),
                        is_training:True
                })  

                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                    z: sample_z(self.batch_size, self.z_dim),
                    is_training:True
                })
    
                if i % self.steps["save"] == 0:
                    saver.save(sess, "tmp/" + self.model_name + ".ckpt")
        
                
                if i % self.steps["sample"] == 0:
                    fake_data = sess.run(gen_data, {
                            z: sample_z(16, self.z_dim),
                            is_training: False
                    })
                    fig = plot_im(fake_data)
                    plt.savefig('out/' + self.model_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print(loss_disc, loss_gen)
            


class WGAN_GP_simple():
    def __init__(self, model_name, batch_size, dataset, num_iter, z_dim,
                 im_dim, steps):
        self.dataset = dataset
        self.num_iter = num_iter
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
    
     def train(self):
        
        tf.reset_default_graph()
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        with tf.variable_scope('generator'):
            z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])    
            #gen_data = archs.generator_conv(z, is_training)
            gen_data = archs.generator_vanilla(z, self.z_dim, self.im_dim)
            
        with tf.variable_scope('discriminator'):
            #x = tf.placeholder(dtype=tf.float32, shape=[None, self.im_dim, self.im_dim], name='x')        
             x = tf.placeholder(\
              tf.float32, shape=[None, self.im_dim*self.im_dim], name='inputs')
           
        #d_real = archs.discriminator_conv(x, self.im_dim, reuse=False)
        #d_fake = archs.discriminator_conv(gen_data, self.im_dim, reuse=True)
        d_real = archs.discriminator_vanilla(x, self.im_dim, reuse=False)
        d_fake = archs.discriminator_vanilla(gen_data, self.im_dim, reuse=True)
        
        
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator/')]
        g_params = [v for v in vars if v.name.startswith('generator/')]
        
        loss_d =  tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        loss_g = - tf.reduce_mean(d_fake)
        
        
        
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty
    
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(-loss_d, var_list=d_params))
            g_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(loss_g, var_list=g_params))
        
        
        clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_params]
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, "tmp/" + self.model_name + ".ckpt")
            except:
                print("Failed to restore")
                init = tf.global_variables_initializer()
                sess.run(init)
            
            for i in range(self.num_iter):
                
                #update discriminator multiple times
                for n_critic in range(5):
                    train_data, _ = self.dataset.next()
                    #train_data = np.reshape(train_data, (-1, 28, 28))
                    
                    loss_disc, _, _ = sess.run([loss_d, d_optimizer, clip_d], {
                            x: train_data,
                            z: sample_z(self.batch_size, self.z_dim),
                            is_training:True
                    })
                    
                #update generator
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                        z: sample_z(self.batch_size, self.z_dim),
                        is_training:True
                })  

    
                if i % self.steps["save"] == 0:
                    saver.save(sess, "tmp/" + self.model_name + ".ckpt")
        
                
                if i % self.steps["sample"] == 0:
                    fake_data = sess.run(gen_data, {
                            z: sample_z(16, self.z_dim),
                            is_training: False
                    })
                    fig = plot_im(fake_data)
                    plt.savefig('out/' + self.model_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print(loss_disc, loss_gen)
                
            
                        
    
        
class WGAN_conv():
    def __init__(self, model_name, batch_size, dataset, num_iter, z_dim,
                 im_dim, steps):
        self.dataset = dataset
        self.num_iter = num_iter
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
        
    def train(self):
        
        tf.reset_default_graph()
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        with tf.variable_scope('generator'):
            z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])    
            gen_data = archs.generator_conv(z, is_training)
    
            
        with tf.variable_scope('discriminator'):
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.im_dim, self.im_dim], name='x')        
                   
        d_real = archs.discriminator_conv(x, self.im_dim, reuse=False)
        d_fake = archs.discriminator_conv(gen_data, self.im_dim, reuse=True)
        
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator/')]
        g_params = [v for v in vars if v.name.startswith('generator/')]
        
        loss_d =  tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        loss_g = - tf.reduce_mean(d_fake)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(-loss_d, var_list=d_params))
            g_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(loss_g, var_list=g_params))
        
        
        clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_params]
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, "tmp/" + self.model_name + ".ckpt")
            except:
                print("Failed to restore")
                init = tf.global_variables_initializer()
                sess.run(init)
            
            for i in range(self.num_iter):
                
                #update discriminator multiple times
                for n_critic in range(5):
                    train_data, _ = self.dataset.next()
                    train_data = np.reshape(train_data, (-1, 28, 28))
                    
                    loss_disc, _, _ = sess.run([loss_d, d_optimizer, clip_d], {
                            x: train_data,
                            z: sample_z(self.batch_size, self.z_dim),
                            is_training:True
                    })
                    
                #update generator
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                        z: sample_z(self.batch_size, self.z_dim),
                        is_training:True
                })  

    
                if i % self.steps["save"] == 0:
                    saver.save(sess, "tmp/" + self.model_name + ".ckpt")
        
                
                if i % self.steps["sample"] == 0:
                    fake_data = sess.run(gen_data, {
                            z: sample_z(16, self.z_dim),
                            is_training: False
                    })
                    fig = plot_im(fake_data)
                    plt.savefig('out/' + self.model_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print(loss_disc, loss_gen)
                
            

       
class WGAN_simple():
    def __init__(self, model_name, batch_size, dataset, num_iter, z_dim,
                 im_dim, steps):
        self.dataset = dataset
        self.num_iter = num_iter
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
        
    def train(self):
        
        tf.reset_default_graph()
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        with tf.variable_scope('generator'):
            z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])    
            #gen_data = archs.generator_conv(z, is_training)
            gen_data = archs.generator_vanilla(z, self.z_dim, self.im_dim)
            
        with tf.variable_scope('discriminator'):
            #x = tf.placeholder(dtype=tf.float32, shape=[None, self.im_dim, self.im_dim], name='x')        
             x = tf.placeholder(\
              tf.float32, shape=[None, self.im_dim*self.im_dim], name='inputs')
           
        #d_real = archs.discriminator_conv(x, self.im_dim, reuse=False)
        #d_fake = archs.discriminator_conv(gen_data, self.im_dim, reuse=True)
        d_real = archs.discriminator_vanilla(x, self.im_dim, reuse=False)
        d_fake = archs.discriminator_vanilla(gen_data, self.im_dim, reuse=True)
        
        
        vars = tf.trainable_variables()
        d_params = [v for v in vars if v.name.startswith('discriminator/')]
        g_params = [v for v in vars if v.name.startswith('generator/')]
        
        loss_d =  tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        loss_g = - tf.reduce_mean(d_fake)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(-loss_d, var_list=d_params))
            g_optimizer = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(loss_g, var_list=g_params))
        
        
        clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_params]
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                saver.restore(sess, "tmp/" + self.model_name + ".ckpt")
            except:
                print("Failed to restore")
                init = tf.global_variables_initializer()
                sess.run(init)
            
            for i in range(self.num_iter):
                
                #update discriminator multiple times
                for n_critic in range(5):
                    train_data, _ = self.dataset.next()
                    #train_data = np.reshape(train_data, (-1, 28, 28))
                    
                    loss_disc, _, _ = sess.run([loss_d, d_optimizer, clip_d], {
                            x: train_data,
                            z: sample_z(self.batch_size, self.z_dim),
                            is_training:True
                    })
                    
                #update generator
                loss_gen, _ = sess.run([loss_g, g_optimizer], {
                        z: sample_z(self.batch_size, self.z_dim),
                        is_training:True
                })  

    
                if i % self.steps["save"] == 0:
                    saver.save(sess, "tmp/" + self.model_name + ".ckpt")
        
                
                if i % self.steps["sample"] == 0:
                    fake_data = sess.run(gen_data, {
                            z: sample_z(16, self.z_dim),
                            is_training: False
                    })
                    fig = plot_im(fake_data)
                    plt.savefig('out/' + self.model_name + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print(loss_disc, loss_gen)
                
            
                        