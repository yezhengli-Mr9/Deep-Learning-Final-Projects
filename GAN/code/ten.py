#import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
import tensorflow as tf




'''
session = tf.Session()

e = [[[1,4],[2,7]],[[3,9],[4,8]]]
t = tf.add(e, 1)
print("INPUT")
print(session.run(tf.concat([e, t], 0)))
ones = tf.ones(5)
zeros = tf.zeros(5)

print(session.run(tf.concat([ones, zeros],0)))


x = tf.constant([1.0])
#y = tf.sigmoid(x)
#print(session.run(y))
labels = tf.constant([1.0])
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=labels)
print(session.run(loss))

segmask_x = tf.constant([[0,1],[0.5,1]])
segmask_car_labels = tf.constant([[0,1.0],[1.0,0]])
segmask_car_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=segmask_x, labels=segmask_car_labels)
print(session.run(segmask_car_loss))
segmask_car_loss = tf.reduce_mean(segmask_car_loss, axis=1)
print(session.run(segmask_car_loss))


name = '/Users/apple/Desktop/Penn2/CIS680/HW3/sample_data/train_data/P&C_dataset_imgs/mask/car/000000.png'
seg_car_path = tf.read_file(name)
seg_car = tf.image.decode_png(seg_car_path)
seg_car = tf.image.resize_images(seg_car, [22, 22])
seg_car = tf.cast(tf.round(seg_car), tf.float32)
print(seg_car.shape)
ones = tf.reduce_sum(tf.cast(tf.equal(seg_car, 1), tf.int32))
zeros = tf.reduce_sum(tf.cast(tf.equal(seg_car, 0), tf.int32))
print(session.run(seg_car))
print(session.run(ones))
print(session.run(zeros))
print(session.run(tf.count_nonzero(seg_car)))

loss = tf.constant([[[0],[0.2],[1]],[[0.3],[0],[0]], [[0.2],[0.1],[0]]])
loss = loss * 255
loss = tf.cast(loss, tf.uint8)
encoded = tf.image.encode_png(loss)
#c = tf.write_file('./out.png', encoded)
#print(session.run(write))
f = open('name.png', "wb+")
f.write(session.run(encoded))
f.close()
'''

'''

A = np.array([[[0],[0.2],[1]],[[0.3],[0],[0]], [[0.2],[0.1],[0]]])
A = 255 - A * 255
A = A.astype(np.uint8)
cv2.imwrite('./here.jpg', A) 
#im = Image.fromarray(A)
#im.save("file.jpg")
'''

'''
A = cv2.imread('/Users/apple/Desktop/Penn2/CIS680/HW3/mini_sample_data/train_data/P&C_dataset_imgs/mask/car/000001.png', cv2.IMREAD_UNCHANGED)
#print(A)
A = 255 - A * 255
A = A.astype(np.uint8)
cv2.imwrite('./here.jpg', A)
'''

x = "Hello World!"
print(len(x))
print(x[2:len(x)])


session = tf.Session()
name = '/Users/apple/Desktop/Penn2/CIS680/HW3/data/celeba_small_sample/imgs/000188.png'
seg_car_path = tf.read_file(name)
seg_car = tf.image.decode_png(seg_car_path, channels=1)
print(seg_car.shape)
print(session.run(seg_car[1]))
#seg_car = tf.image.resize_images(seg_car, [22, 22])
#seg_car = tf.cast(tf.round(seg_car), tf.float32)
#print(seg_car.shape)
#ones = tf.reduce_sum(tf.cast(tf.equal(seg_car, 1), tf.int32))
#zeros = tf.reduce_sum(tf.cast(tf.equal(seg_car, 0), tf.int32))
#print(session.run(seg_car))
#print(session.run(ones))
#print(session.run(zeros))
#print(session.run(tf.count_nonzero(seg_car)))