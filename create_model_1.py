# Copyright 2017 Ranveer SIngh.
#
# ==============================================================================


'''
In this tutorials we will train our model at basic level, and try to see whether its able to understand random selected image from the training model or not, after that , we will test it with the custom image 28*28 pixels.
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os.path
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

FLAGS = None

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def run_training():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

# Define loss and optimizer, GradientDescentOptimizer is of the most used training algorithm
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    init_op=tf.global_variables_initializer()
    saver = tf.train.Saver()
    tempData=""
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(200)
            print(batch_xs)
            print(batch_ys)
            
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Storing the model at defined folder
        save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
        print ("Model saved in file: ", save_path)
        # Genrating random number between 0 to 55000, for selecting one of the 55000 trained image
        #display_random(random.randint(0,55000),sess,y,x)
        # For testing custom image 
        display_custom(sess,y,x)
        
    '''
    This method will be yoused for passing the custom image and testing , 
    whether trained model able to verify the passed custom image or not
    '''
def display_custom(sess,y,x):
    images = np.zeros((1,784))
    '''
    Setting Label as 4 because, we are passing 42.png which is an image of  4.
    make sure, you pass image of 28*28 pixels, with Black background and white number
    '''
    label=4
    gray = cv2.imread("img/41.png", 0 ) #0=cv2.CV_LOAD_IMAGE_GRAYSCALE #must be .png!

    # rescale it
    gray = cv2.resize(gray, (28, 28))

    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    images[0] = flatten

    my_classification = sess.run(tf.argmax(y, 1), feed_dict={x: [images[0]]})

    """
    We want to show the predicted value and the actual value along with graph. matplotlib 
    library is usefull for plotting the graph.
    """
    plt.title('Prediction: %d Label: %d' % (my_classification, label))
    plt.imshow(images[0].reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    print('Neural Network predicted', my_classification[0], "for your digit")
    plt.show()
    
        
        
    '''
    This method will be used for testing image form MNIST using random genarted number.
    '''
def display_random(num,sess,y,x):
    # THIS WILL LOAD ONE TRAINING EXAMPLE
    x_train = mnist.train.images[num,:].reshape(1,784)
    y_train = mnist.train.labels[num,:]
    # THIS GETS OUR LABEL AS A INTEGER
    label = y_train.argmax()
    print(x_train)
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

def main(_):
    '''
    Creating the flolder structure, if not present else deldting structure and creating folder again.
    '''
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    log_dir this will be used to create a directroey in your system, 
    where trained model will get stored
    '''
    parser.add_argument(
      '--log_dir',
      type=str,
      default='/Model1/tensorflow/mnist',
      help='Directory to put the log data.'
  )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
