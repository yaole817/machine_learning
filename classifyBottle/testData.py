import src.kNN as knn
import src.img as imgLib
import cv2
import os
import sys
import numpy as np
import trainData

dir_path = '.\\res'
fileList =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']
path = dir_path+'\\'+fileList[int(sys.argv[1])]

def testData():
	train_datas=[]
	lables=[]
	for item in fileList:
		train_data,lable = trainData.loadImages(dir_path+'\\'+item)
		train_datas.extend(train_data)
		lables.extend(lable)
	imgData,imgLable = trainData.loadImages(dir_path+'\\'+fileList[0])
	print(knn.classify0(imgData[0],train_datas,lables,3))


if __name__ == '__main__':
	testData()