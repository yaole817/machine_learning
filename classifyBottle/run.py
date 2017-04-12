#!/usr/bin/python
#-*-coding:utf-8 -*-
import src.img as imgLib
import cv2
import os
import sys
import numpy as np
dir_path = '.\\res'
list =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']
path = dir_path+'\\'+list[int(sys.argv[1])]

if __name__ == '__main__':
    # 读取文件
    #imagePath = sys.argv[1]
    #img = cv2.imread('./res/xue7.jpg')
    #detect(img)
    img = cv2.imread(path)
    #img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    '''img = np.array(img)         
    mean = np.mean(img)         
    img = img - mean         
    img = img*1.5 + mean*1.5 #修对比度和亮度         
    img = img/255.
    cv2.imshow('pic',img)
    cv2.waitKey()
    '''
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
    imageBlue   = imgLib.findBottleCap(image)

    image = imgLib.colorDetect(img,1)
    imageYellow = imgLib.findBottleCap(image)
    #image = imgLib.findTextRegion(image)
    imageYellow.extend(imageBlue)
    
    for box in imageYellow:
        #cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        
        cutImg= imgLib.cutImg(box,img)
        gray = cv2.cvtColor(cutImg,cv2.COLOR_BGR2GRAY) 
        gray = cutImg[:,:,1]
        imgLib.cutImageFromCircle(gray)
        #gray=cutImg[:,:,0]
        #sharpImg = imgLib.SuanSharp(cutImg)
        #gray = cv2.cvtColor(cutImg, cv2.COLOR_BGR2GRAY)
        #textImage = imgLib.extractTextOutline(gray)
        '''
        hist= cv2.calcHist([gray], [0], None, [256], [0.0,255.0])
        max_pixel_num = max(hist)
        max_pixel = 0
        for i in range(len(hist)):
            if hist[i] == max_pixel_num:
                max_pixel = i
                break
        imgSize = gray.shape
        hight = imgSize[0]
        width = imgSize[1]
        max_pixel = gray[hight/10][width/2]
        
        binaryImg = imgLib.binarizateImage(gray,max_pixel-3,max_pixel+3)
        #binaryImg = imgLib.sharpImg(gray,165,185)
        #cv2.imwrite("binary.jpg", sharpImg)
        '''
        imgLib.grayHist(gray)
        cv2.imshow('img',gray)
        
        #imgLib.grayHist(gray)
        cv2.waitKey(0)
    #print(region)
    #imgLib.showImage(img)



