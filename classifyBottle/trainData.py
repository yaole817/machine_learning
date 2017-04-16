import src.kNN as knn
import src.img as imgLib
import cv2
import os
import sys
import numpy as np
dir_path = '.\\database'
fileList =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']
path = dir_path+'\\'+fileList[int(sys.argv[1])]

def img2Vect(img,lable):
    image = imgLib.colorDetect(img,0) # detect the blue colour
    imageBlue   = imgLib.findBottleCap(image)  # find the bottle area in one picture

    image = imgLib.colorDetect(img,1) # detect yellow colour
    imageYellow = imgLib.findBottleCap(image)

    imageYellow.extend(imageBlue)
    trainData = []
    lableData = []
    i = 0
    for box in imageYellow:
        cutImg= imgLib.cutImg(box,img)
        cutImg = cv2.resize(cutImg,(600,600))
        cv2.imwrite(".\database\\"+lable+str(i)+'.jpg',cutImg)  # write img to database
        i+=1
        gray = cv2.cvtColor(cutImg,cv2.COLOR_BGR2GRAY)  # convert img to gray img
        gray = cutImg[:,:,1] 
        imgLib.cutImageFromCircle(gray)   
        hist = cv2.calcHist([gray], [0], None, [256], [1.0,255.0]) # calculate the hist of img
        trainData.append(hist)
        lableData.append(lable)
    return trainData,lable

def loadImages(filepath):

    lable = filepath.split('\\')[-1].replace('.jpg','')[:-2]

    img = cv2.imread(filepath)
    trainData,lable = img2Vect(img,lable)
    return trainData,lable 

def gray2GrayHist(gray):

    hist = cv2.calcHist([gray], [0], None, [256], [1.0,255.0])
    hist = np.int0(hist)
    grayHistArray = hist[:,0]

    return grayHistArray 

def getLabel(name):

    return name.split('\\')[-1].replace('.jpg','')[:-2]

def creatDataSet(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    imgLib.cutImageFromCircle(gray)  
    gray_hist_array = gray2GrayHist(img)
    lable = getLabel(path)
    return gray_hist_array,lable


def loadDataSet(fileList):
    dataSets=[]
    lables = []
    for item in fileList:
        gray_hist_array,lable = creatDataSet(item)
        dataSets.append(gray_hist_array)
        lables.append(lable)
    return dataSets,lables


if __name__ == '__main__':
    
    fileLists = [dir_path+'\\'+item for item in fileList]
    testData,testlable= creatDataSet('.\database\changshengBlue1.jpg')
    dataSets,lables=loadDataSet(fileLists)
    result = knn.classify0(testData,dataSets,lables,3)

    print result
    print testlable
    exit()
    trainDatas = []
    lables = []
    for item in fileList:
        trainData,lable = loadImages(dir_path+'\\'+item)
        trainDatas.extend(trainData)
        lables.extend(lable)
    with open('dataSet.txt','w') as f:
        f.write(''.join(str(trainDatas)))
        f.write('\n'.join(lables))
        #cv2.imshow('img',gray)
        
        #imgLib.grayHist(gray)
        #cv2.waitKey(0)
