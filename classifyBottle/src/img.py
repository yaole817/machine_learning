#!/usr/bin/python
#-*-coding:utf-8 -*-
import sys
import cv2
import numpy as np
import random

def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) #30,9
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) #24,36

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    # 7. 存储中间图片 
    #cv2.imwrite("binary.png", binary)
    #cv2.imwrite("dilation.png", dilation)
    #cv2.imwrite("erosion.png", erosion)
    #cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    derp,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print contours
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        #print cnt

        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 
        # 面积小的都筛选掉
        if(area < 10000):
            continue
        
        
        # 轮廓近似，作用很小
        '''epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        '''
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print "rect is: "
        print rect

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)


         #计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        #if(width <height * 1.2):
         #   continue
        
        region.append(box)

    return region


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    # 带轮廓的图片
    cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imageGray(path):
    
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    return image

def binarizateImage(image,min_num,max_unm):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]>min_num and image[i][j]<max_unm:
                image[i][j] = 255
            else:
                image[i][j] =0
    return image


def findBottleCap(img):
    region = []

    # 1. 查找轮廓
    derp,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print contours
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        #print cnt

        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 
        # 面积小的都筛选掉
        if(area < 1000):
            continue
        
        
        # 轮廓近似，作用很小
        '''epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        '''
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        
       #print "rect is: "
        #print rect

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)


         #计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        #if(width <height * 1.2):
         #   continue
        
        region.append(box)

    return region

def colorDetect(image,option=0):
    name = random.randint(0,99)
    img = image
    colorImage = img.copy()
    #_colorImage = img.copy()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv)
    #高斯模糊
    img = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow("hsv",hsv)
    # 设定蓝色的阈值
    if(option == 0):
        lower=np.array([100,50,50])
        upper=np.array([140,255,255])
    else:
        #黄色
        lower=np.array([15,100,100])
        upper=np.array([40,255,255])

    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower,upper)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(img,img,mask=mask)
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #二值化
    ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('gray',gray)
    #闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3))  
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)  
    #cv2.imshow('closed',closed)
    '''
    _,cnts,_ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = cnts
    #cv2.drawContours(img,cnts,-1,(0,0,255),1)
    imgRs = []
    i = 0
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        if(w<50 or h < 15 or w>h < 1.0):
            continue
        #cv2.rectangle(_colorImage,(x,y),(x+w,y+h),(0,255,0),1)
        #imgCrop = _colorImage[y:y+h,x:x+w]
        box = cv2.boxPoints(cnt)
        box = np.int0(box)
        imgRs.append(box)
        rs = img[y:y+h,x:x+w]
        #cv2.imshow("============="+str(name),rs)

    #cv2.drawContours(_colorImage, [_box], -1, (0,0,255), 1)
    #cv2.imshow("_colorImage",_colorImage)
    '''
    return closed
def showImage(image):
    #for i in range(len(image)):
    #    print image[i]
    cv2.imshow("img",image)
    cv2.waitKey(0)
