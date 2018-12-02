import tensorflow as tf
from layers import conv_factory
from layers import fc_factory
from layers import transpose_conv_factory



def build_gan_generator(z, batch_size, is_train, reuse):
  with tf.variable_scope('generator', reuse=reuse) as vs:
    x = z
    
    with tf.variable_scope('fc', reuse=reuse):
      hidden_num = 4 * 4 * 1024      
      x = fc_factory(x, hidden_num, is_train, reuse, with_rec=False)
      x = tf.reshape(x,[batch_size, 4, 4, 1024])
      
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 512
      x = transpose_conv_factory(x, hidden_num, 5, 2, is_train, reuse) 
    
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num = 256
      x = transpose_conv_factory(x, hidden_num, 5, 2, is_train, reuse) 
      
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 128
      x = transpose_conv_factory(x, hidden_num, 5, 2, is_train, reuse) 
    
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 1
      x = transpose_conv_factory(x, hidden_num, 5, 2, is_train, reuse, with_rec=False) 

    x = tf.nn.tanh(x)
    print('generate imgs shape: ', x.shape)
    generated_imgs = x
              
  generator_vars = tf.contrib.framework.get_variables(vs)
        
  return generated_imgs, generator_vars


def build_gan_discriminator(x, batch_size, is_train, reuse):
  with tf.variable_scope('discriminator', reuse=reuse) as vs:
      
    x = x + tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.1, dtype=tf.float32)     
     
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 128
      x = conv_factory(x, hidden_num, 5, 2, is_train, reuse, with_rec=True, activation='leaky_relu') 
    
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num = 256
      x = conv_factory(x, hidden_num, 5, 2, is_train, reuse, with_rec=True, activation='leaky_relu') 
      
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 512
      x = conv_factory(x, hidden_num, 5, 2, is_train, reuse, with_rec=True, activation='leaky_relu') 
    
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 1024
      x = conv_factory(x, hidden_num, 5, 2, is_train, reuse, with_rec=True, activation='leaky_relu') 
     
    x = tf.layers.flatten(x)       
    print('discriminator procedure x shape after flattern: ', x.shape)
     
    with tf.variable_scope('fc', reuse=reuse):
      hidden_num = 1     
      x = fc_factory(x, hidden_num, is_train, reuse, with_rec=False)
      
    discriminator_x = x
    print('discriminator_x shape: ', discriminator_x.shape)
      
  discriminator_vars = tf.contrib.framework.get_variables(vs)
        
  return discriminator_x, discriminator_vars


def compute_loss_accuracy(x, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=labels)
    loss = tf.reduce_mean(loss) 
      
    accuracy = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(x)), labels), tf.float32)
    accuracy = tf.reduce_mean(accuracy) 
    
    return loss,accuracy


def build_gan(x, batch_size, is_train, reuse):
    true_imgs = x
    z = tf.random_uniform([batch_size, 100], minval=-1.0,maxval=1.0, dtype=tf.float32)
        
    true_labels = tf.ones((batch_size, 1))
    generated_labels = tf.zeros((batch_size, 1))
    
    generated_imgs, generator_vars = build_gan_generator(z, batch_size, is_train, reuse)
    discriminator_score_true, discriminator_vars_true = build_gan_discriminator(true_imgs, batch_size, is_train, reuse)
    discriminator_score_generated, discriminator_vars_generated = build_gan_discriminator(generated_imgs, batch_size, is_train, True)
    
    generator_loss, generator_accuracy = compute_loss_accuracy(discriminator_score_generated, true_labels)   
    discriminator_loss_true, discriminator_accuracy_true = compute_loss_accuracy(discriminator_score_true, true_labels) 
    discriminator_loss_generated, discriminator_accuracy_generated = compute_loss_accuracy(discriminator_score_generated, generated_labels) 
    
    discriminator_loss = (discriminator_loss_true + discriminator_loss_generated) / 2.0
    discriminator_accuracy = (discriminator_accuracy_true + discriminator_accuracy_generated)/2.0
    loss_list = [generator_loss, discriminator_loss]
    weighted_loss = tf.reduce_mean(loss_list)
    
    return weighted_loss, loss_list, [generated_imgs, discriminator_score_true, discriminator_score_generated], [generator_accuracy, discriminator_accuracy], [generator_vars,discriminator_vars_true, discriminator_vars_generated]


def create_model(model_type, x, labels_list, c_num, batch_size, is_train, reuse):
    print('create model: ', model_type)
    if (model_type == 'gan'):
        return build_gan(x, batch_size, is_train, reuse)

    