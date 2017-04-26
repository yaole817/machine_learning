#!/usr/bin/python
#-*-coding:utf-8 -*-
import sys
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

class BaseImg:
    def __init__(self,img,imgName='_'):
        self.img  = img
        self.imgName = imgName
        self.gray = []
        self.colorImgBox = []
        self.cutImg = []
        self.dataSet = []
        self.lable = []    # differ to other object


        
    def colorDetect(self,image,option=0):
        # belong to the color of the image detect the interest areas
        # 
        name = random.randint(0,99)
        img = image.copy()
        
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
        #cv2.imshow('gray',gray)
        #闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3))  
        closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)  
        return closed


    def findBottleCap(self,img):
        boxes = []

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
           # rect = cv2.boundingRect()
            
           #print "rect is: "
            #print rect

            # box是四个点的坐标
            box = cv2.boxPoints(rect)
        # if box[0][0] != box[1][0]: # img has angle
        # give to the boxes,which have angle, cut the minmum interest Area
        # the algorithm is descirbed as:
        # 1. suppose the boxes is a square 
        # 2. get the centel of the boxes,calculate the catercorner of square
            imgCenter  = (box[2] - box[0])/2+box[0]
            imgHalfLen = np.int((sum((box[2] - box[0])**2)**0.5)/2.8) # cal the lenth of circle
            imgNewYMax = np.int(imgCenter[1]+imgHalfLen)
            imgNewYMin = np.int(imgCenter[1]-imgHalfLen)
            imgNewXMax = np.int(imgCenter[0]+imgHalfLen)
            imgNewXMin = np.int(imgCenter[0]-imgHalfLen)

            box[0][0] = box[1][0]  =  imgNewXMin
            box[2][0] = box[3][0]  =  imgNewXMax
            box[0][1] = box[3][1]  =  imgNewYMax
            box[1][1] = box[2][1]  =  imgNewYMin
            if imgNewXMin < 0 or  imgNewXMax >img.shape[1] or imgNewYMin < 0 or imgNewYMax>img.shape[0]:
                continue
            box = np.int0(box)
            boxes.append(box)
        return boxes

    def blueAndYellowColorDetect(self,img):

        blueImg    = self.colorDetect(img,0)
        blueImgBox = self.findBottleCap(blueImg) 
        self.colorImgBox.extend(blueImgBox)

        yellowImg  = self.colorDetect(img,1)
        yellowImgBox = self.findBottleCap(yellowImg) 
        self.colorImgBox.extend(yellowImgBox)

        return self.colorImgBox


    def cutImgFrombox(self,boxes,img):
        # if box[0][0] != box[1][0]: # img has angle
        # give to the boxes,which have angle, cut the minmum interest Area
        # the algorithm is descirbed as:
        # 1. suppose the boxes is a square 
        # 2. get the centel of the boxes,calculate the catercorner of square
        # 3. calculate the length of square,and the points
        for box in boxes:
            imgNewXMin = box[0][0] 
            imgNewXMax = box[2][0]
            imgNewYMax = box[0][1]
            imgNewYMin = box[1][1]

            cut_img   = img[imgNewYMin:imgNewYMax, imgNewXMin:imgNewXMax]
            cut_img = cv2.resize(cut_img,(600,600))
            self.cutImg.append(cut_img)
        return self.cutImg

    def cutGrayFromCircle(self,cutImg): 
        #  the interest ares is round, if we want improve the accuracy rate,
        #  the best method is to cut the outside of the circle
        #  the input cutImg is squal image from one big image

        gray = cv2.cvtColor(cutImg,cv2.COLOR_BGR2GRAY) 
        imgCenter  = [len(gray)/2,len(gray[0])/2]
        r = int(len(gray)*4/5)
        for i in range(len(gray)):
            for j in range(len(gray[i])):
                if ((imgCenter[0]-i)**2+(imgCenter[0]-j)**2)>r**2:
                    gray[i][j] = 0
        return gray

    def gray2GrayHist(self,gray):
        # calculate the hist of one img belong to the gray image

        hist = cv2.calcHist([gray], [0], None, [256], [1.0,255.0])
        hist = np.int0(hist)
        grayHistArray = hist[:,0]

        return grayHistArray  

    def getLable(self):
        # set lable of one Image from the image name
        lable = self.imgName.split('\\')[-1]
        index = lable.index('_')

        return lable[:index]

    def createDataSets(self): 
        # create date set from one img
        #  
        colorImgBoxes = self.blueAndYellowColorDetect(self.img)
        cutImg  = self.cutImgFrombox(colorImgBoxes,self.img)
        if cutImg[0]==[]:
            cutImg.pop(0)
        for item in cutImg:
            #gray = self.cutGrayFromCircle(item)
            gray = cv2.cvtColor(item,cv2.COLOR_BGR2GRAY) 
            gray_hist = self.gray2GrayHist(gray)
            self.dataSet.append(gray_hist)
            self.lable.append(self.getLable())
        return self.dataSet

    def saveImg(self,img,name):
        cv2.imwrite(name,img)


def grayHist(gray):
    # display hist of One image due to gray image
    hist= cv2.calcHist([gray], [0], None, [256], [1.0,255.0])
    plt.figure()#新建一个图像
    plt.title("Grayscale Histogram")#图像的标题
    plt.xlabel("Bins")#X轴标签
    plt.ylabel("# of Pixels")#Y轴标签
    plt.plot(hist)#画图
    plt.xlim([0,256])#设置x坐标轴范围
    plt.show()#显示图像

def showImage(image):
    #for i in range(len(image)):
    #    print image[i]
    cv2.imshow("img",image)
    cv2.waitKey(0)

def loadDataSet(path):
    img = cv2.imread(path)
    baseImg = BaseImg(img,path)

    return baseImg.dataSet,baseImg.lable

if __name__ == '__main__':
    img_name = 'changshengBlue_0.jpg'
    img = cv2.imread(img_name)
    baseImg = BaseImg(img,img_name)
    print baseImg.createDataSets()
    print baseImg.lable
    print baseImg.centers
