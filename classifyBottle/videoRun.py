import numpy as np
import cv2
import src.img as imgLib
import src.imgFilter as imgFilter
import os
import src.kNN as knn


dir_path = '.\\res'
video_path = dir_path+'\\'+'test.mp4'

img_dirPath = '.\\database'
imgPathList =  [img_dirPath+'\\'+x for x in os.listdir(img_dirPath) if os.path.splitext(x)[1]=='.jpg']


if __name__ == '__main__':
    dataSets = []
    dataLables = []
    '''
        this code is for load database
        create the db
    '''
    for item in imgPathList:
        print item
        img = cv2.imread(item)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        baseImg = imgFilter.BaseImg(img,item)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grayHistArray = baseImg.gray2GrayHist(gray)
        lable = baseImg.getLable()
        dataSets.append(grayHistArray)
        dataLables.append(lable)

    '''
        this is for test img
    '''
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        videoImg = imgFilter.BaseImg(frame)
        testSet  = videoImg.createDataSets()
        
        for i in range(len(videoImg.colorImgBox)-1):
            box = videoImg.colorImgBox[i]
            test_img = testSet[i]

            result_lable = knn.classify0(test_img,dataSets,dataLables,5)

            cv2.drawContours(frame, [box], 0, (0, 255, 0), 4)
            imgCenter  = (box[2] - box[0])/2+box[0]
            cv2.putText(frame,result_lable, (imgCenter[0],imgCenter[1]),cv2.FONT_HERSHEY_SIMPLEX,1.4, (0, 0, 255),4)
        
        frame = cv2.resize(videoImg.img, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
