import tensorflow as tf
import numpy as np
import time
import os
from ops import *
from utils import *
from glob import glob

class SNGAN:
    model_name = 'SNGAN'
    
    def __init__(self, input_height=64, input_width=64, input_channels=1, output_height=64, output_width=64, gf_dim=64, df_dim=64, batchsize=64, z_dim = 100, is_crop=False, learning_rate=0.0002  , beta1=0.5, beta2=0.999, input_fname_pattern = '*.jpg', is_grayscale=False, dataset_name = 'celebA', checkpoint_dir = './checkpoint', sample_dir = 'sample', epoch = 30, sess=None):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.input_fname_pattern = input_fname_pattern      
        self.is_grayscale = is_grayscale
        self.is_crop = is_crop
        
        self.output_height = output_height
        self.output_width = output_width
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.batchsize = batchsize
        self.z_dim = z_dim
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        
        self.disc_iters = 1
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.sess = sess

    def generator(self, noise_z, is_training=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            # auto-encoder structure
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            #fc1_bn = tf.nn.relu(batch_norm(linear(noise_z, 1024, scope_name='g_fc1'), is_training=is_training, name='g_fc1_bn'))
            fc2 = linear(noise_z, s_h16*s_w16*self.gf_dim*8, scope_name='g_fc2')
            fc2_reshape = tf.reshape(fc2, [-1, s_h16, s_w16, self.gf_dim*8])
            fc2_deconv = tf.nn.relu(batch_norm(fc2_reshape, is_training=is_training, name='g_fc2_bn'))
            print("deconv2d_1:", fc2_deconv)
            # deconv layer_2
            filter_shape2 = [5, 5, self.gf_dim*4, self.gf_dim*8]
            output_shape2 = [self.batchsize, s_h8, s_w8, self.gf_dim*4]
            h_deconv2 = tf.nn.relu(batch_norm(deconv2d(fc2_deconv, filter_shape2, output_shape2, scope_name='g_deconv2'), is_training=is_training, name='g_bn_deconv2'))
            print("deconv2d_2:",h_deconv2)
            
            # deconv layer_3
            filter_shape3 = [5,5,self.gf_dim*2, self.gf_dim*4]
            output_shape3 = [self.batchsize, s_h4, s_w4, self.gf_dim*2]
            h_deconv3 = tf.nn.relu(batch_norm(deconv2d(h_deconv2, filter_shape3,output_shape3, scope_name='g_deconv3'), is_training=is_training, name='g_bn_deconv3'))
            print("deconv2d_3:", h_deconv3)
            
            # deconv layer_4
            filter_shape4 = [5,5,self.gf_dim, self.gf_dim*2]
            output_shape4 = [self.batchsize, s_h2, s_w2, self.gf_dim]
            h_deconv4 = tf.nn.relu(batch_norm(deconv2d(h_deconv3, filter_shape4,output_shape4, scope_name='g_deconv4'), is_training=is_training, name='g_bn_deconv4'))
            print("deconv2d_4:", h_deconv4)
            
            # deconv layer_5
            filter_shape5 = [5,5,self.input_channels, self.gf_dim]
            output_shape5 = [self.batchsize, s_h, s_w, self.input_channels]
            h_deconv5 = tf.nn.tanh(deconv2d(h_deconv4, filter_shape5, output_shape5, scope_name='g_deconv5'))
            print("deconv2d_5:", h_deconv5)
            
            return h_deconv5

    def discriminator(self, input_data_x, is_training=True, reuse=False, update_collection=None):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            # discriminator, cnn structure
            # shape is the size of the filter
            # hidden layer_1
            shape1 = [5, 5, self.input_channels, self.df_dim]
            shape2 = [5, 5, self.df_dim, self.df_dim*2]
            shape3 = [5, 5, self.df_dim*2, self.df_dim*4]
            shape4 = [5, 5, self.df_dim*4, self.df_dim*8]
            # hidden layer_2                        
            h_conv1 = leaky_relu(batch_norm(conv2d(input_data_x, shape1, scope_name='d_conv1', spectral_norm=True, update_collection=update_collection), is_training=is_training, name='d_bn_conv1'))
            print("h_conv2_1:", h_conv1)
            
            # hidden layer_2            
            h_conv2 = leaky_relu(batch_norm(conv2d(h_conv1, shape2, scope_name='d_conv2', spectral_norm=True, update_collection=update_collection), is_training=is_training, name='d_bn_conv2'))
            print("h_conv2_2:", h_conv2)
            
            # hidden layer_3
            h_conv3 = leaky_relu(batch_norm(conv2d(h_conv2, shape3, scope_name='d_conv3', spectral_norm=True, update_collection=update_collection), is_training=is_training, name='d_bn_conv3'))
            print("h_conv2_3", h_conv3)

            h_conv4 = leaky_relu(batch_norm(conv2d(h_conv3, shape4, scope_name='d_conv4', spectral_norm=True, update_collection=update_collection), is_training=is_training, name='d_bn_conv4'))
            print("h_conv2_4", h_conv4)            

            h_conv4_flat = tf.reshape(h_conv4, [self.batchsize, -1])
            # hidden layer_4 fully connected
            h_fc1_logits = linear(h_conv4_flat, 1024, scope_name='d_fc1', spectral_norm=True, update_collection=update_collection)
            
            return h_fc1_logits

    def build_model(self):
        # crop image
        if self.is_crop:
            img_dims = [self.input_height, self.input_width, self.input_channels]
        else:
            img_dims = [self.input_height, self.input_width, self.input_channels]
            
        self.input_data = tf.placeholder(tf.float32, [self.batchsize] + img_dims, name='real_data')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='noise')
        
        # real images
        self.D_real = self.discriminator(self.input_data, is_training=True, reuse=False, update_collection=None)
        # fake images
        self.G_sample = self.generator(self.z, is_training=True, reuse=False)
        self.D_fake = self.discriminator(self.G_sample, is_training=True, reuse=True, update_collection='NO_OPS')
        # sample images
        
        # cross entrophy
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))
        
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake, tf.ones_like(self.D_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        # save model
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        #self.d_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        #self.g_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        self.d_optimization = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimization = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=self.g_vars)

        self.sample_images = self.generator(self.z, is_training=False, reuse=True)
        # saver for saving model
        self.saver = tf.train.Saver()

    def train(self):        
        try:
            tf.global_variables_initializer().run()
        except AttributeError:
            tf.initialize_all_variables().run()
        # sample real_images and noise_z for testing
        sample_data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        print(len(sample_data))
        sample_files = sample_data[0:self.batchsize]
        sample_batch_x = [get_image(sample_file,is_grayscale=self.is_grayscale) for sample_file in sample_files]
        if (self.is_grayscale):
            sample_batch_x = np.array(sample_batch_x).astype(np.float32)[:, :, :, None]
        else:
            sample_batch_x = np.array(sample_batch_x).astype(np.float32)

        sample_z = np.random.uniform(-1,1, [self.batchsize, self.z_dim]).astype(np.float32)
        sample_batch_x = 2*((sample_batch_x/255.)-.5)

        counter_bool, counter = self.load(self.checkpoint_dir)
        if counter_bool:
            counter = counter + 1
            print("[***]load model successfully")
        else:
            counter = 1
            print("[***]fail to load model")
        start_time = time.time()
        for index in range(self.epoch):
            # code just for images datasets
            data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            batch_idxs = int(len(data)/self.batchsize)
            
            for idx in range(batch_idxs):
                batch_files = data[idx*self.batchsize:(idx+1)*self.batchsize]
                # load data from datasets
                batch = [get_image(batch_file, is_grayscale=self.is_grayscale) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_x = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_x = np.array(batch).astype(np.float32)
                # normalization
                batch_x = 2*((batch_x/255.)-.5)
                batch_z = np.random.uniform(-1,1, [self.batchsize, self.z_dim]).astype(np.float32)
                
                # update discriminator
                _ = self.sess.run(self.d_optimization, feed_dict={self.input_data:batch_x, self.z:batch_z})

                # update generator again
                _ = self.sess.run(self.g_optimization, feed_dict={self.z:batch_z})
                _ = self.sess.run(self.g_optimization, feed_dict={self.z:batch_z})
                
                # calc loss
                d_loss = self.sess.run(self.d_loss, feed_dict={self.input_data:batch_x,
                                                               self.z:batch_z})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.z:batch_z})
                iteration_time = time.time()
                total_time = (iteration_time - start_time)
                print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:", total_time, "d_loss:",  d_loss,"g_loss:", g_loss)
                
                counter = counter + 1
                if np.mod(idx, 100) == 0:
                    iteration_time = time.time()
                    total_time = (iteration_time - start_time)
                    # sample images and save them
                    samples = self.sess.run(self.sample_images, feed_dict={self.z:sample_z})
                    #print(samples)
                    save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, index, idx))
                    # calc loss
                    d_loss_ = self.sess.run(self.d_loss, feed_dict={self.input_data:sample_batch_x,
                                                               self.z:sample_z})
                    g_loss_ = self.sess.run(self.g_loss, feed_dict={self.z:sample_z})
                    print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:", total_time, "d_loss:",  d_loss_,"g_loss:", g_loss_)
                # save model
                if np.mod(counter, 500) == 0:
                    self.save_model(self.checkpoint_dir, counter)
    
    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batchsize, self.z_dim)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
