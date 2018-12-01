from __future__ import print_function                                                                 


from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np  
import os                                                                                  
import cv2



def create_folder(parent_folder, config):
   time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
   file_type = config.model_type + '_' + config.dataset + '_' + str(config.batch_size) + '_' + str(config.max_step) + '_'
   folder_path = parent_folder + '/' + file_type + time_str
   os.mkdir(folder_path)
   return folder_path
   
def save_generated_imgs(output_generated_imgs, generated_imgs_folder, step):
    for i in range(output_generated_imgs.shape[0]):
        img = output_generated_imgs[i]
        out_filename = generated_imgs_folder + '/step' + str(step) + '_imgid' + str(i) + '.png'
        #img= img.astype(np.uint8)
        cv2.imwrite(out_filename, img)

def save_plot(output_training_loss, loss_plot_folder, step):
  out_filename = loss_plot_folder + '/step' + str(step) + '.png'
  ys = np.asarray(output_training_loss, dtype=np.float32)
  iter_steps = ys.shape[0]
  n = ys.shape[1]
  x = np.arange(iter_steps)
  for i in range(n):
    plt.plot(x, ys[:,i])
  legend = ['generator loss', 'discriminator loss']
  legend = legend[0:n]
  plt.legend(legend)
  plt.title('Training Loss')
  plt.savefig(out_filename)