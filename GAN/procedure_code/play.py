 

from datetime import datetime
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import os


def plot_multiple_y(input_ys):
  ys = np.asarray(input_ys, dtype=np.float32)
  iter_steps = ys.shape[0]
  n = ys.shape[1]
  x = np.arange(iter_steps)
  for i in range(n):
    plt.plot(x, ys[:,i])
  #plt.plot(x, 2 * x)
  #plt.plot(x, 3 * x)
  #plt.plot(x, 4 * x)
  legend = ['rpn classification loss', 'rpn regression loss', 'object classification loss', 'binary mask loss']
  legend = legend[0:n]
  plt.legend(legend)
  #plt.legend(legend, loc='upper left')
  plt.title('Training Loss')
  plt.savefig('./result_plot/multiple_result.jpg')
  plt.show()

def plot_y():
  train_accuracy = [[9],[10],[11]]
  plt.plot(np.arange(len(train_accuracy)), train_accuracy)
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  plt.title('Training accuracy')
  plt.savefig('./result_plot/result1.jpg')
  print("Train accuracy: ", train_accuracy[-1], "%")
  plt.show()
  #print("Test accuracy : ", in_training_test_accuracy, "%")

#itemlist = [4.2, 56.5]
#time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#folder = './result'
#file_type = '/test'
#out_filename = folder + file_type + time_str

#print(np.arange(3))
#

#c = [[1,2,3],[4,5,6],[7,8,9]]
#array = np.asarray(c, dtype=np.float32)
#plot_multiple_y(c)



#A = 255 - A * 255
#A = A.astype(np.uint8)




    
A = np.array([[[249.2],[0.1]],[[0],[255]]])
cv2.imwrite('test1.png', A)
print(A[-1])

#create_folder('procedure_generated_imgs')


 
  