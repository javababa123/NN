
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import os.path
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as pl


def bacthing():
    # step 1
    filenames = ['a.png', 'b.png', 'c.png', 'd.png']

    # step 2
    filename_queue = tf.train.string_input_producer(filenames)

    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_jpeg(content, channels=3)
    image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(image, [224, 224])

    # step 4: Batching  
    image_batch = tf.train.batch([resized_image], batch_size=8)
    print(filename)


def processBreakImage():
    for foldername in os.listdir('/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data'):
        folderstr=os.path.join('/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data',foldername)
        tempfolder=os.path.join('/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data2',foldername)
        tf.gfile.MakeDirs(os.path.join('/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data2',foldername))
        count=1
        for filename in os.listdir(folderstr):
            newImage = Image.new('L', (28, 28),"white")
            digit = Image.open(os.path.join(folderstr,filename)).convert('L')   
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
            pixel=digit.load() 
            for x in range(int(width)):
                temp=False
                for y in range(int(height)):
                    if(pixel[x,y]>127):
                        pixel[x,y]=255
                    else:
                        pixel[x,y]=0
            
            if foldername in('I','ii','J','jj','L','ll','tt','1','ff'):
                size =digit.size
                heightratio= 26/size[1]
                reduced_size = int(size[0] * heightratio), int(size[1] * heightratio)      
                digit1 = digit.resize(reduced_size, Image.ANTIALIAS)
                size =digit1.size
                height=size[1]
                width =size[0]
                x= (28-width)/2
                y= (28-height)/2
                print(width,",",height)
                newImage.paste(digit1, (int(x),int(y)))   
                newImage.save(os.path.join(tempfolder,str(count)+".png"), "PNG")
                count=count+1
                
            else :    
                digit= digit.resize((26, 26))   
                newImage.paste(digit, (1,1))        
                newImage.save(os.path.join(tempfolder,str(count)+".png"), "PNG")
                count=count+1
                
    
def qualityImage():
    im = Image.open("a.png").convert('L') 
    size =im.size   # get the size of the input image
    ratio = 10  # reduced the size to 90% of the input image
    reduced_size = int(size[0] * ratio), int(size[1] * ratio)     

    im_resized = im.resize(reduced_size, Image.ANTIALIAS)
    im_resized.save("a2.png", "PNG")
    processImage(im_resized)
    
    

def processImage(newImage):
    width = float(newImage.size[0])
    height = float(newImage.size[1])
    pixel=newImage.load() 
    print(pixel)
    for x in range(int(width)):
        temp=False
        for y in range(int(height)):
            if(pixel[x,y]>60):
                pixel[x,y]=255
            else:
                pixel[x,y]=0
    pl.imshow(newImage)
    pl.show()
    return newImage

def breakimage(hcharacter,im,page):    
    oldX=0
    oldY=0
    width = float(im.size[0])  
    row=1
    column=1
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
            
            
            #image1 =image.crop((32,0,21+32,20))
            #pl.imshow(image)
            #pl.show()
            
            for j in range(len(wcharacter)-1):  
                if(((int(wcharacter[j+1])-int(wcharacter[j]))>2)):
                    diff=(int(wcharacter[j+1])-int(wcharacter[j]))
                    image1 =image.crop((int(wcharacter[j]),0,(int(wcharacter[j])+diff),height1))
                    
                    filename=str(page)+"-"+str(row)+"-"+str(column)
                    filename=filename+".png"
                    tf.gfile.MakeDirs(("/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data1/"+str(column)))
                    image1.save(os.path.join("/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/data1/"+str(column),filename ), "PNG", quality=100)
                    print(str(row)+":"+str(column))
                    if(column>61):
                        column=0
                        row=row+1
                    column=column+1
                    

def startBreakImage():
    page=1
    im = Image.open(os.path.join("/Self Driving Car/How to Use Tensorboard (LIVE)/JPG-PNG-to-MNIST-NN-Format-master/scan/","D1.png")).convert('L')
    size =im.size   # get the size of the input image
    ratio = 2  # reduced the size to 90% of the input image
    reduced_size = int(size[0] * ratio), int(size[1] * ratio)     
    im = im.resize(reduced_size, Image.ANTIALIAS)
    im=processImage(im)        
    size =im.size   # get the size of the input image
    width = float(im.size[0])
    height = float(im.size[1])
    hcharacter=[]
    for x in range(int(height)):
        temp=False
        for y in range(int(width)):
            pixel=im.load()[y,x]
            if(pixel==0):
                temp=True
        if(temp==False):
            hcharacter.append(str(x))
    #print(hcharacter)
    breakimage(hcharacter,im,page)
    page=page+1
    
                    
                    
def main():
    #processBreakImage();
    im = Image.open("java.jpg").convert('L')
    processImage(im)
    
if __name__ == "__main__":
    main()