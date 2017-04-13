import src.kNN as knn
import src.img as imgLib
import cv2
import os
import sys
import numpy as np
dir_path = '.\\res'
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
    for box in imageYellow:
        cutImg= imgLib.cutImg(box,img)
        gray = cv2.cvtColor(cutImg,cv2.COLOR_BGR2GRAY) 
        gray = cutImg[:,:,1]
        imgLib.cutImageFromCircle(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [1.0,255.0])
        trainData.append(hist)
        lableData.append(lable)
    return trainData,lable

def loadImages(filepath):

    lable = filepath.split('\\')[-1].replace('.jpg','')[:-2]

    img = cv2.imread(filepath)
    trainData,lable = img2Vect(img,lable)
    print lable
    return trainData,lable 

if __name__ == '__main__':
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
