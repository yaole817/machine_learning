import numpy as np
import cv2
import src.img as imgLib
dir_path = '.\\res'
path = dir_path+'\\'+'test.mp4'
cap = cv2.VideoCapture(path)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    image = imgLib.colorDetect(frame,0)
    imageBlue   = imgLib.findBottleCap(image)

    image = imgLib.colorDetect(frame,1)
    imageYellow = imgLib.findBottleCap(image)
    #image = imgLib.findTextRegion(image)
    imageYellow.extend(imageBlue)

    for box in imageBlue:
        cv2.drawContours(gray, [box], 0, (0, 255, 0), 2)
    cv2.imshow('frame',gray)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()