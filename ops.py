import tensorflow as tf

# convolution
def conv2d(input_x, kernel_size, stride=[1,2,2,1], scope_name='conv2d', conv_type='SAME', spectral_norm=True, update_collection=None):
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection)

        conv = tf.nn.bias_add(tf.nn.conv2d(input_x, weights, strides=stride, padding=conv_type), bias)
        return(conv)

# deconvolution
def deconv2d(input_x, kernel_size, output_shape, stride=[1,2,2,1], scope_name='deconv2d', deconv_type='SAME'):
    output_len = kernel_size[2]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', kernel_size, tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        try:
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        except:
            deconv = tf.nn.bias_add(tf.nn.deconv2d(input_x, weights, output_shape, strides=stride, padding=deconv_type), bias)
        return deconv

# batch normalization
def batch_norm(input_x, epsilon=1e-5, momentum=0.9, is_training = True, name='batch_name'):
    with tf.variable_scope(name) as scope:
        batch_normalization = tf.contrib.layers.batch_norm(input_x,
                                              decay=momentum,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              scale=True,
                                              is_training=is_training,
                                              scope=name)
        return batch_normalization
        
# fully connected
def linear(input_x, output_size, scope_name='linear', spectral_norm=True, update_collection=None):
    shape = input_x.get_shape()
    input_size = shape[1]
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights', [input_size, output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        

        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection)
        output = tf.matmul(input_x, weights) + bias
        return output

# leaky_relu
def leaky_relu(input_x, leaky=0.2):
    return tf.maximum(leaky*input_x, input_x)

# pooling
def max_pool(input_data_x, filter_shape=[1,2,2,1], pooling_type='SAME'):
    if pooling_type == 'SAME':
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)
    else:
        return tf.nn.max_pool(input_data_x, ksize=filter_shape, strides=[1,2,2,1], padding=pooling_type)

# spectral_norm
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm

def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1
        
        u_hat, v_hat,_ = power_iteration(u,iteration)
        
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        
        w_mat = w_mat/sigma
        
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not(update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))
            
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm
