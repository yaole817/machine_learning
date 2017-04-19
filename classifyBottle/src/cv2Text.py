
import sys
import cv2
import numpy as np

img = cv2.imread('changshengBlue_0.jpg')
cv2.putText(img,"Hello World !", (200,200),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255),2)
cv2.imshow("img",img)
cv2.waitKey(0)
