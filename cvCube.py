import numpy as np
import cv2
cap = cv2.VideoCapture(0)
import math


counter = 0
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
    lower = np.array([80, 35, 130])
    upper = np.array([180, 255, 240])
  
    mask = cv2.inRange(hsv, lower, upper)
      
    result = cv2.bitwise_and(frame, frame, mask = mask)

    ret,result = cv2.threshold(result,90,255,0)
  
    cv2.imshow('result', result)
    if cv2.waitKey(1)== ord('q'):
        break

  
cv2.destroyAllWindows()
cap.release()