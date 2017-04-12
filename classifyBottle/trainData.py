import src.kNN as knn
import src.img as imgLib
import cv2
import os
import sys
import numpy as np
dir_path = '.\\res'
list =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']
path = dir_path+'\\'+list[int(sys.argv[1])]

if __name__ == '__main__':

    img = cv2.imread(path)

    imgR=img[:,:,0]
    imgG=img[:,:,1]
    imgB=img[:,:,2]
    image = imgG-imgB
    
    #image = imageGray(path)
    image = imgG
    
    image = imgLib.colorDetect(img,0)
    imageBlue   = imgLib.findBottleCap(image)

    image = imgLib.colorDetect(img,1)
    imageYellow = imgLib.findBottleCap(image)
    #image = imgLib.findTextRegion(image)
    imageYellow.extend(imageBlue)
    trainData=[]
    for box in imageYellow:
        cutImg= imgLib.cutImg(box,img)
        gray = cv2.cvtColor(cutImg,cv2.COLOR_BGR2GRAY) 
        gray = cutImg[:,:,1]
        imgLib.cutImageFromCircle(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [1.0,255.0])
        trainData.append(hist)
        #cv2.imshow('img',gray)
        
        #imgLib.grayHist(gray)
        #cv2.waitKey(0)
    with open('trainData.txt','w') as f:
    	for i in trainData:
    		f.write(str(i))
    		f.write('\n')
