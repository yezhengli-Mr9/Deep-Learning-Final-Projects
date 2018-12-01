import numpy as np                                                                                    
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


def save_result(results, config): 
  time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
  folder = './result'
  file_type = '/' + config.model_type + '_' + config.dataset + '_' + str(config.batch_size) + '_' + str(config.max_step) + '_'
  out_filename = folder + file_type + time_str
  with open(out_filename, 'wb') as fp:
    pickle.dump(results, fp)

def load_result(filetime):
  folder = './result'
  file_type = '/test'
  input_filename = folder + file_type + filetime
  with open (input_filename, 'rb') as fp:
    results = pickle.load(fp) 
    return results

def plot_multiple_y(input_ys, config):
  time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
  folder = './result_plot'
  file_type = '/' + config.model_type + '_' + config.dataset + '_' + str(config.max_step) + '_'
  out_filename = folder + file_type + time_str + '.png'
  ys = np.asarray(input_ys, dtype=np.float32)
  iter_steps = ys.shape[0]
  n = ys.shape[1]
  x = np.arange(iter_steps)
  for i in range(n):
    plt.plot(x, ys[:,i])
  legend = ['generator loss', 'discriminator loss']
  legend = legend[0:n]
  plt.legend(legend)
  #plt.legend(legend, loc='upper left')
  plt.title('Training Loss')
  plt.savefig(out_filename)
  plt.show()

def plot_result(results, config):
  train_accuracy = results['train_accuracy']
  train_weighted_loss = results['train_weighted_loss']
  train_all_loss = results['train_all_loss']
  
  #test_accuracy = results['test_accuracy']
  #test_weighted_loss = results['test_weighted_loss']
  #test_all_loss = results['test_all_loss']
  
  print(' ')
  print('----------final result----------')
  print('train_accuracy: ', train_accuracy)
  print('train_weighted_loss: ', train_weighted_loss)
  print('train_all_loss: ', train_all_loss)
  #print('test_accuracy: ', test_accuracy)
  #print('test_weighted_loss: ', test_weighted_loss)
  #print('test_all_loss: ', test_all_loss)

  plot_multiple_y(train_all_loss, config)

def main(config):  
  prepare_dirs_and_logger(config)

  tf.set_random_seed(config.random_seed)

  train_data_loader, train_label_loader_list = get_loader(
    config.dataset, config.data_path, config.batch_size, 0, 'train', True)
  '''
  if config.is_train:
    test_data_loader, test_label_loader_list = get_loader(
      config.dataset, config.data_path, config.batch_size_test, config.testset_start_index, 'test', False)
  else:
    test_data_loader, test_label_loader_list = get_loader(
      config.dataset, config.data_path, config.batch_size_test, 0, config.split, False)
   '''

  test_data_loader = None
  test_label_loader_list = []
  trainer = Trainer(config, train_data_loader, train_label_loader_list, test_data_loader, test_label_loader_list)
  if config.is_train:
    save_config(config)
    results = trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    #trainer.test()
      
  save_result(results, config)
  
  #results = load_result('2018_11_17_13_53_10')
  plot_result(results, config)
  
   
  

if __name__ == "__main__":
  config, unparsed = get_config()
  main(config)
