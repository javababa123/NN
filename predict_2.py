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
import constant as ct


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


def imageprepare(argv):
    im = argv
    tv = list(im.getdata())
    tva = [ (x)*1.0/255.0 for x in tv]
    return tva


def processBreakImage(image1,prediction,keep_prob,sess,x1,finalOutput):
    newImage = Image.new('L', (28, 28),"white")
    digit = image1
    width = float(digit.size[0])
    height = float(digit.size[1])
    top=-1
    bottom=0
    left=-1
    right=0   
    for x in range(int(height)):
        for y in range(int(width)):
            temp=digit.load()[y,x]
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
    digit = digit.crop((left, top,right+1,bottom+1))
    width = float(digit.size[0])
    height = float(digit.size[1])
    size =digit.size
    ratio=height   
    if(height<width):     
        ratio=width   
    heightratio= 26/ratio
    reduced_size = int(size[0] * heightratio), int(size[1] * heightratio)      
    digit1 = digit.resize(reduced_size, Image.ANTIALIAS)
    pixel=digit1.load() 
    for x in range(int(digit1.size[0])):
        temp=False
        for y in range(int(digit1.size[1])):
            if(pixel[x,y]>220):
                pixel[x,y]=255
            else:
                pixel[x,y]=0
    
    size =digit1.size
    
    height=size[1]
    width =size[0]
    x= (28-width)/2
    y= (28-height)/2
    newImage.paste(digit1, (int(x),int(y)))
    #pl.imshow(newImage)
    #pl.show()
    predictValue= prediction.eval(feed_dict={x1: [imageprepare(newImage)],keep_prob: 1.0}, session=sess)
    #print(chr(int(ct.mapping[str(predictValue[0])])))
    return chr(int(ct.mapping[str(predictValue[0])]))
    


def breakimage(hcharacter,im,page,prediction,keep_prob,sess,x1):    
    oldX=0
    oldY=0
    width = float(im.size[0])  
    row=1
    column=1
    finalOutput=""
    for i in range(len(hcharacter)-1):
        if(((int(hcharacter[i+1])-int(hcharacter[i]))>1)):
            wcharacter=[]
            wide=(int(hcharacter[i+1]))
            oldY= int(hcharacter[i])
            image =im.crop((oldX,oldY,im.size[0],wide))
       
            width1 = float(image.size[0])
            height1 = float(image.size[1])
            for x in range(int(width1)):
                temp=False
                for y in range(int(height1)):
                    pixel=image.load()[x,y]       
                    if(pixel==0):
                        temp=True
                if(temp==False):
                    wcharacter.append(str(x))
       
            for j in range(len(wcharacter)-1):  
                if(((int(wcharacter[j+1])-int(wcharacter[j]))>1)):
                    diff=(int(wcharacter[j+1])-int(wcharacter[j]))
                    image1 =image.crop((int(wcharacter[j]),0,(int(wcharacter[j])+diff),height1))
                    finalOutput=finalOutput+processBreakImage(image1,prediction,keep_prob,sess,x1,finalOutput)
    print(finalOutput)

def startBreakImage(argv,prediction,keep_prob,sess,x1):
    page=1
    im = Image.open(argv).convert('L')
    size =im.size   # get the size of the input image
    heightratio=2
    reduced_size = int(size[0] * heightratio), int(size[1] * heightratio)      
    im = im.resize(reduced_size, Image.ANTIALIAS)
    pl.imshow(im)
    pl.show()
    size =im.size   # get the size of the input image
    width = float(im.size[0])
    height = float(im.size[1])
    
    hcharacter=[]
    
    pixel=im.load() 
    for x in range(int(width)):
        temp=False
        for y in range(int(height)):
            if(pixel[x,y]>127):
                pixel[x,y]=255
            else:
                pixel[x,y]=0
    
    for x in range(int(height)):
        temp=False
        for y in range(int(width)):
            pixel=im.load()[y,x]
            if(pixel==0):
                temp=True
        if(temp==False):
            hcharacter.append(str(x))
    breakimage(hcharacter,im,page,prediction,keep_prob,sess,x1)
    page=page+1


    
def main(argv):
    """ 
    Calling processImage for making the passed image clear i.e either 255 or 0 &
    imagebreakup is for identifying the different digits in the given image
    """
    # Define the model (same as when creating the model file)
    x1 = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 62]))
    b = tf.Variable(tf.zeros([62]))
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x1, [-1,28,28,1])
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
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    '''
    This part will call the actual algorithm for predicting all the value  present in the given image,
    Here we are resoting model saved in previous tutorial.
    '''
    with tf.Session() as sess:  
        sess.run(init_op)
        saver.restore(sess, os.path.join("/ReadIT/tensorflow/mnist", 'model.ckpt'))
        prediction=tf.argmax(y_conv,1)
        startBreakImage(argv,prediction,keep_prob,sess,x1)
    '''
    imvalue = imageprepare(argv)
    #print(imvalue)
    predictValue= prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
    #print(predictValue)
    #print(chr(int(ct.mapping[str(predictValue[0])])))
    '''
if __name__ == "__main__":
    main(sys.argv[1])
