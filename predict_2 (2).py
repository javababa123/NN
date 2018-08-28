# Copyright 2017 Ranveer Singh. 
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


import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import os.path
import numpy as np
import matplotlib.pyplot as pl
import webbrowser


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
       
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   


def imageprepare(argv,oldX,oldY,wide):
    """
    This function will crop th imageinto individual image and resize to 20*20 pixel image,
    Then it will pase the image to 28*28 pixel image.
    """
    im = Image.open(argv).convert('L')
    im =im.crop((oldY,oldX,wide,im.size[1]))
    width = float(im.size[0])
    height = float(im.size[1])
    top=-1
    bottom=0
    left=-1
    right=0    
    newImage = Image.new('L', (28, 28),"white")
    for x in range(int(height)):
        for y in range(int(width)):
            temp=im.load()[y,x]
            if(temp==0):    
                if(top==-1 and top<=x):
                    top=x
                    #print("T:",top)
                if((left==-1 and left<y) or ( left>y)):
                    left=y
                    #print("L:",left)
                if(bottom<x):
                    bottom=x
                    #print("B:",bottom)
                if(right < y):
                    right=y
                    #print("R:",right)
                
    print("top=",top,",left=",left,",right=",right+1,",bottom=",bottom+1)
    
    im = im.crop((left, top,right+1,bottom+1))
    width1= right+1-left
    height1= bottom+1-top
    #pl.imshow(im)
    #pl.show()
    #print(width1,height1)
    bigger=height1
    if(width1>height1):
        
        bigger=width1
        imagePixel = Image.new('L', (bigger,bigger),"white")
        imagePixel.paste(im, (0,int(bigger/2)-int(height1/2)))
        imagePixel= imagePixel.resize((20, 20))
        #x, y = im.size
        newImage.paste(imagePixel, (4,4))
        '''
        #newImage.save("test.jpg", "JPEG")
        # optional, show the saved image in the default viewer (works in Windows)
        #webbrowser.open("test.jpg")
        '''
        pl.imshow(newImage)
        pl.show()
        tv = list(newImage.getdata())
        #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [ (255-x)*1.0/255.0 for x in tv]
        return tva
    elif(width1<height1):
        print("h")
        imagePixel = Image.new('L', (bigger,bigger),"white")
        imagePixel.paste(im, ((int(bigger/2)-int(width1/2),0)))
        imagePixel= imagePixel.resize((20, 20))
        newImage.paste(imagePixel, (4,4))
        #pl.imshow(newImage)
        #pl.show()
        '''
        #newImage.save("test.jpg", "JPEG")
        # optional, show the saved image in the default viewer (works in Windows)
        #webbrowser.open("test.jpg")
        '''
        pl.imshow(newImage)
        pl.show()
        tv = list(newImage.getdata())
        #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [ (255-x)*1.0/255.0 for x in tv]
        return tva
        
# This method will determine the breaking point of given image into individual digits.
def imagebreakup(argv):
    im = argv
    width = float(im.size[0])
    height = float(im.size[1])
    character=[]
    for x in range(int(width)):
        temp=False
        for y in range(int(height)):
            pixel=im.load()[x,y]
            if(pixel==0):
                temp=True
        if(temp==False):
            character.append(str(x))
    print(character)
    return character

def processImage(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    pixel=im.load()       
    for x in range(int(width)):
        temp=False
        for y in range(int(height)):
            if(pixel[x,y]>127):
                pixel[x,y]=255
            else:
                pixel[x,y]=0
    return im;
    
def main(argv):
    """ 
    Calling processImage for making the passed image clear i.e either 255 or 0 &
    imagebreakup is for identifying the different digits in the given image
    """
    imbreakup=imagebreakup(processImage(argv))
    length = len(imbreakup)
   
    oldX=0
    oldY=0
    finalData=[]
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
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
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    '''
    This part will call the actual algorithm for predicting all the value  present in the given image,
    Here we are resoting model saved in previous tutorial.
    '''
    finalData=[]
    with tf.Session() as sess:  
        sess.run(init_op)
        sess.run(W_conv1)
        saver.restore(sess, os.path.join("/Model2/tensorflow/mnist", 'model.ckpt'))
        prediction=tf.argmax(y_conv,1)
        for i in range(len(imbreakup)-1):
            if(((int(imbreakup[i+1])-int(imbreakup[i]))>1)):
                print("row: ",i)
                imvalue = imageprepare(argv,oldX,oldY,int(imbreakup[i+1]))
                
                predictValue= prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
                finalData.append(predictValue)
                oldY=int(imbreakup[i+1])
    for i in range(len(finalData)):
        print(finalData[i])
    
    
if __name__ == "__main__":
    main(sys.argv[1])
