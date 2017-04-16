import cv2
import numpy as np

img = cv2.imread('./database/changshengyellow1.jpg')
#img = img[int(img.shape[0]*0.146):int(img.shape[0]*0.854),int(img.shape[1]*0.146):int(img.shape[1]*0.854)]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('original',img)
#cv2.waitKey()
'''
corners = cv2.goodFeaturesToTrack(gray,200,0.05,1)
print(corners)
corners = np.int0(corners)
print(corners)
for i in corners:
	x,y = i.ravel()  
	cv2.circle(img,(x,y),3,255,-1)
cv2.imshow('SURF',img)
cv2.waitKey()
'''

kernel = np.array([ [-1, -1, -1],  
                    [-1,  12, -1],  
                    [-1, -1, -1] ])  
  
dst = cv2.filter2D(img, -1, kernel) 

gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

cv2.imshow('dst', gray)
cv2.waitKey()