# Copyright 2017 Ranveer Singh.
# Adapted form the on the MNIST expert tutorial by Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#import modules
import tensorflow as tf
import argparse
from PIL import Image, ImageFilter
import numpy as np
import argparse
import os.path
import sys
import constant as ct
import scipy.ndimage
import time
import matplotlib.pyplot as pl
FLAGS = None
#import data from MNIST

sess = tf.InteractiveSession()



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

'''
createModel will create the model for training and testing numbers.
'''
def createModel():                        
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 62])
    W = tf.Variable(tf.zeros([784, 62]))
    b = tf.Variable(tf.zeros([62]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 62])
    b_fc2 = bias_variable([62])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Define loss and optimizer
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    """
    Train the model and save the model to disk as a model.ckpt file
    file is stored in the same directory as this python script is started
    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    startindex=0  
    for i in range(2500):
        x_train=[]
        y_label=[]
        count=21
        for foldername in os.listdir(FLAGS.img_dir):
            folderstr=os.path.join(FLAGS.img_dir,foldername)
            for filename in os.listdir(folderstr):
                intfilename=filename[0: filename.find(".") ]
                if (int(intfilename)>=(startindex) and int(intfilename)<(startindex+count)):
                    #print(os.path.join(folderstr,filename))
                    digit = Image.open(os.path.join(folderstr,filename)).convert('L')
                    tv = list(digit.getdata())
                    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
                    tva = [ (x)*1.0/255.0 for x in tv]
                    x_train.append(np.array(tva).reshape(-1, 784))
                    imageName="v"+str(foldername)
                    value= ct.values[imageName]
                    y_label.append(np.array(value).astype(np.float))      
        if(startindex>=183):
            startindex=0  
        else:
            startindex=startindex+count
        print(i)
        x_train = np.array(x_train).reshape(-1, 784)
        y_label=np.array(y_label)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:x_train, y_: y_label, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x:x_train, y_: y_label, keep_prob: 0.5})
    
    # Storing the model at defined folder
    save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    print ("Model saved in file: ", save_path)
        
def main(_):
    '''
    Creating the flolder structure, if not present else deldting structure and creating folder again.
    '''
    #if tf.gfile.Exists(FLAGS.log_dir):
        #tf.gfile.DeleteRecursively(FLAGS.log_dir)
    #tf.gfile.MakeDirs(FLAGS.log_dir)
    createModel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    log_dir this will be used to create a directroey in your system, 
    where trained model will get stored
    '''
    parser.add_argument(
      '--log_dir',
      type=str,
      default='/ReadIT/tensorflow/mnist',
      help='Directory to put the log data.'
  )
    parser.add_argument(
      '--img_dir',
      type=str,
      default='/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data2',
      help='Directory to put the log data.'
  )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
