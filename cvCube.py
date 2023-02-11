import numpy as np
import cv2
cap = cv2.VideoCapture(0)
import math


counter = 0
import time

otime = time.time()

while(True):
    counter+=1
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 5)
      
    lower = np.array([114, 33, 60])
    upper = np.array([180, 255, 255])
  
    mask = cv2.inRange(hsv, lower, upper)
      
    result = cv2.bitwise_and(frame, frame, mask = mask)

    ret,result = cv2.threshold(result,1,255,0)

    copy_result = result

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.medianBlur(result, 5)

    np_result = np.array(result)
    np_result[np_result != 0] = 255
    result = cv2.merge((np_result, np_result, np_result))

    cv2.GaussianBlur(result, (51, 51), 0)
    kernel = np.ones((5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.medianBlur(result, 5)
    result = cv2.medianBlur(result, 5)

    result = cv2.Canny(result, 80, 120, -1)

    kernel = np.ones((1, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(np.array(result), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    approx_contours = sorted(approx_contours, key=lambda c: cv2.contourArea(c), reverse=True)
    if approx_contours:
        approx_contours = [approx_contours[0]]

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))


    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    cones = []
    bounding_rects = []
    b2 = []
    for ch in convex_hulls_3to10:
        cones.append(ch)
        rect = cv2.boundingRect(ch)
        bounding_rects.append(rect)
        b2.append(cv2.minAreaRect(ch))

    if b2:
        (zx, zy), (width, height), angle  = b2[0]
        if width/height >= 0.75 and width/height <= 1.25 and width*height>=2500:
            print("DETECTION: " + str(width) + "," + str(height) + " | " + str(zx) + "," + str(zy))
            pass
 
    if counter % 30 == 0:
        counter = 0
        print("FPS: " + str(30/(time.time()-otime)))
        otime = time.time()

        #copy_result = cv2.drawContours(copy_result,[np.int0(cv2.boxPoints(b2[0]))],0,(0,0,255),2)

    #copy_result = cv2.putText(copy_result, "ENABLED: "+str(0), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    #cv2.imshow('RES', copy_result)

    if cv2.waitKey(1)== ord('q'):
        break

  
cv2.destroyAllWindows()
cap.release()