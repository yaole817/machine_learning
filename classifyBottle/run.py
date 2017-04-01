#!/usr/bin/python
#-*-coding:utf-8 -*-
import src.img as imgLib
import cv2
import os
import sys
dir_path = '.\\res'
list =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']
path = dir_path+'\\'+list[int(sys.argv[1])]


if __name__ == '__main__':
    # 读取文件
    #imagePath = sys.argv[1]
    #img = cv2.imread('./res/xue7.jpg')
    #detect(img)
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    imgR=img[:,:,0]
    imgG=img[:,:,1]
    imgB=img[:,:,2]
    image = imgG-imgB
    
    #image = imageGray(path)
    image = imgG
    
    #showImage(image)
    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    #image = binarizateImage(image,5,60)
    #image = imgLib.preprocess(image)
    #image = cv2.blur(image,(3,3))
    #imgLib.showImage(image)
    image = imgLib.colorDetect(img,0)
    image = imgLib.findBottleCap(image)
    
  
    
    for box in image:
        print(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    #print(region)
    imgLib.showImage(img)
