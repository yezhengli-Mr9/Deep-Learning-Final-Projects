from __future__ import print_function                                                                 

import tensorflow as tf
import sys
import numpy as np
from tqdm import trange
import cv2

from models import create_model

from output_helper import create_folder
from output_helper import save_generated_imgs
from output_helper import save_plot


def norm_img(img):
  return img / 127.5 - 1.

def denorm_img(img):
  return (img + 1.) * 127.5

class Trainer(object):
  def __init__(self, config, data_loader, label_loader_list, test_data_loader, test_label_loader_list):
    self.config = config
    self.data_loader = data_loader
    self.label_loader_list = label_loader_list
    self.test_data_loader = test_data_loader
    self.test_label_loader_list = test_label_loader_list

    self.optimizer = config.optimizer
    self.batch_size = config.batch_size
    self.batch_size_test = config.batch_size_test

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.img_save_step = config.img_save_step
    self.save_step = config.save_step
    self.test_iter = config.test_iter
    self.wd_ratio = config.wd_ratio
    self.model_type = config.model_type

    self.lr = tf.Variable(config.lr, name='lr')
    self.generator_lr =  0.0002
    self.discriminator_lr = 0.0002

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    
    self.generated_imgs_folder = create_folder('procedure_generated_imgs', config)    
    self.loss_plot_folder = create_folder('procedure_loss_plot', config) 
    
    self.build_model()
    #self.build_test_model()

    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)


  def sum_list(self, listA, listB):
    return [x + y for x, y in zip(listA, listB)]
    
  def divide_list(self, listA, dividend):
    for i in range((len(listA))):
        listA[i] = listA[i] / (dividend * 1.0)
    return listA

  def save_img(self,A, name):
      A = 255 - A * 255
      A = A.astype(np.uint8)
      cv2.imwrite(name, A)

  def train(self):

    #test_accuracy = 0.0
    train_accuracy = []
    train_weighted_loss = []
    train_all_loss = []

    for step in trange(self.start_step, self.max_step):
      #print('!!!!!!!!step:', step)
      fetch_dict = {
        'generator_optim': self.generator_optim,
        'discriminator_optim': self.discriminator_optim,
        'all_loss': self.all_loss,
        'weighted_loss': self.weighted_loss,
        'accuracy': self.accuracy,
        'feat':self.feat}

      if step % self.log_step == self.log_step - 1:
        fetch_dict.update({
          'lr': self.lr,
          'summary': self.summary_op })

      result = self.sess.run(fetch_dict)
      
      train_accuracy.append(result['accuracy'])
      train_all_loss.append(result['all_loss'])
      train_weighted_loss.append(result['weighted_loss'])

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()

        lr = result['lr']
        weighted_loss = result['weighted_loss']
        all_loss = result['all_loss']
        accuracy = result['accuracy']

        print(step, self.max_step, lr, weighted_loss, accuracy)
        print("\n[{}/{}:{:.6f}] Weighted Loss: {:.6f} All Loss {:} Accuracy: {:}" . \
              format(step, self.max_step, lr, weighted_loss, all_loss, accuracy))
        sys.stdout.flush()
        
      if step % self.img_save_step == self.img_save_step - 1:

          output_generated_imgs = denorm_img(result['feat'][0])
          save_generated_imgs(output_generated_imgs, self.generated_imgs_folder, step)
          
          output_all_loss = train_all_loss
          save_plot(output_all_loss, self.loss_plot_folder, step)
          
          
      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model')
        
        '''
        test_accuracy = [0.0 for x in range(len(self.test_accuracy))]
        test_weighted_loss = 0.0
        test_all_loss = [0 for x in range(len(self.test_all_loss))]
        for iter in xrange(self.test_iter):
          fetch_dict = { "test_accuracy":self.test_accuracy, 
          "test_weighted_loss":self.test_weighted_loss,
          "test_all_loss":self.test_all_loss,
          "test_feat": self.test_feat}
          result = self.sess.run(fetch_dict)
          test_accuracy = self.sum_list(test_accuracy, result['test_accuracy'])
          test_weighted_loss = test_weighted_loss + result['test_weighted_loss']
          test_all_loss = self.sum_list(test_all_loss, result['test_all_loss'])
          
        test_accuracy = self.divide_list(test_accuracy, self.test_iter)
        test_weighted_loss = test_weighted_loss / (self.test_iter * 1.0)
        test_all_loss = self.divide_list(test_all_loss, self.test_iter)

        print("\n[{}/{}:{:.6f}] Test Accuracy: {:} Test Loss: {:.4f} Test All Loss: {:}" . \
              format(step, self.max_step, lr, test_accuracy, test_weighted_loss, test_all_loss))
        '''
        sys.stdout.flush()

      #if step % self.epoch_step == self.epoch_step - 1:
       # self.sess.run([self.lr_update])
        
    #results = {'train_accuracy' : train_accuracy, 'train_weighted_loss' : train_weighted_loss, 'train_all_loss' : train_all_loss, 
    #'test_accuracy': test_accuracy, 'test_weighted_loss' : test_weighted_loss, 'test_all_loss' : test_all_loss}
        
    results = {'train_accuracy' : train_accuracy, 'train_weighted_loss' : train_weighted_loss, 'train_all_loss' : train_all_loss}
    return results


  def build_model(self):
    self.x = self.data_loader
    self.labels = self.label_loader_list
    x = norm_img(self.x)

    #self.c_loss, feat, self.accuracy, self.c_var = quick_cnn(
      #x, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False)

    #self.rpn_c_loss, self.rpn_c_feat, self.rpn_c_accuracy, self.rpn_c_all_vars, self.total_count = build_rpn_classifier(x, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False)
    #self.c_loss = self.rpn_c_loss
    #self.weighted_loss = self.rpn_c_loss
    #self.accuracy = self.rpn_c_accuracy
    #self.all_vars = self.rpn_c_all_vars
    
    self.weighted_loss, self.all_loss, self.feat, self.accuracy, self.all_vars = create_model(self.model_type, x, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False)        
    print('before optimizer')
    
    generator_loss = self.all_loss[0]
    discriminator_loss = self.all_loss[1] 
    generator_vars = self.all_vars[0]
    discriminator_vars_true = self.all_vars[1]
    discriminator_vars_generated = self.all_vars[2]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
       self.generator_optim = tf.train.AdamOptimizer(learning_rate=self.generator_lr, beta1=0.5).minimize(generator_loss, var_list=generator_vars)
       self.discriminator_optim = tf.train.AdamOptimizer(learning_rate=self.discriminator_lr, beta1=0.5).minimize(discriminator_loss, var_list=(discriminator_vars_true, discriminator_vars_generated))

    self.summary_op = tf.summary.merge([
      #tf.summary.scalar("c_loss", self.rpn_c_loss),
      #tf.summary.scalar("accuracy", self.rpn_c_accuracy),
      tf.summary.scalar("lr", self.lr),

      #tf.summary.image("inputs", self.x),

      #tf.summary.histogram("feature", self.rpn_c_feat)
    ])


  #def test(self):
    '''
    self.saver.restore(self.sess, self.model_dir)
    test_accuracy = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_accuracy":self.test_accuracy}
      result = self.sess.run(fetch_dict)
      test_accuracy += result['test_accuracy']
    test_accuracy /= self.test_iter

    print("Accuracy: {:.4f}" . format(test_accuracy))
    '''
  
  def build_test_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader_list
    test_x = norm_img(self.test_x)
    #self.test_rpn_c_loss, self.test_rpn_c_feat, self.test_rpn_c_accuracy, feature_vars, intermediate_vars, rpn_classifier_vars, total_count = build_rpn_classifier(test_x, self.test_labels, self.c_num, self.batch_size, is_train=False, reuse=True)
    #self.test_accuracy = self.test_rpn_c_accuracy
    
    
    #self.test_rpn_c_loss, self.test_rpn_c_feat, self.test_rpn_c_accuracy, feature_all_vars, total_count = build_rpn_classifier(test_x, self.test_labels, self.c_num, self.batch_size, is_train=False, reuse=True)
    #self.test_accuracy = self.test_rpn_c_accuracy

    self.test_weighted_loss, self.test_all_loss, self.test_feat, self.test_accuracy, test_all_vars = create_model(self.model_type, test_x, self.test_labels, self.c_num, self.batch_size, is_train=False, reuse=True)    
    
    '''
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader_list
    test_x = norm_img(self.test_x)

    loss, self.test_feat, self.test_accuracy, var = quick_cnn(
      test_x, self.test_labels, self.c_num, self.batch_size_test, is_train=False, reuse=True)
     '''
