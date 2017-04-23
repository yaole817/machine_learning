import numpy as np
import cv2
import src.img as imgLib
import src.imgFilter as imgFilter
import os


dir_path = '.\\res'
video_path = dir_path+'\\'+'test.mp4'

img_dirPath = '.\\database'
imgPathList =  [x for x in os.listdir(img_dirPath) if os.path.splitext(x)[1]=='.jpg']


if __name__ == '__main__':
    dataSets = []
    dataLables = []

    for item in imgPathList:
        print item
        dataset,lable = imgFilter.loadDataSet(item)
        dataSets.append(dataset)
        dataLables.append(lable)


    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        videoImg = imgFilter.BaseImg(frame)
        testSet  = videoImg.createDataSets()
        for box in videoImg.colorImgBox:
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            imgCenter  = (box[2] - box[0])/2+box[0]
            cv2.putText(frame,"Hello World !", (imgCenter[0],imgCenter[1]),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255),2)
        
        cv2.imshow('frame',videoImg.img)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
