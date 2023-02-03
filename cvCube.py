import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
import math

purple = np.uint8([[[211, 63, 293]]])


lower_yellow_e = np.array([250, 209, 255])
upper_yellow_e = np.array([93, 63, 111])


if not cap.isOpened():
    print("error")
    exit()
counter = 430
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow_e, upper_yellow_e)
    result = cv.bitwise_and(frame, frame, mask = mask)

    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    copy_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    ret,thresh = cv.threshold(copy_result,30,255,0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    QAREA = 0
    zx = 0
    zy = 0

    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
        if len(approx) == 3 or len(approx) == 4 or len(approx) == 5  or len(approx) == 6:
            x, y, w, h = cv.boundingRect(cnt)
            if w*h > QAREA:
                QAREA = h*h
                zx = x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                zy = y
 
            ratio = float(w)/h

    if QAREA > 0:
        if 325000/QAREA < 1000:
            #if counter%10==0:
                #cv.imwrite("./TFM/images/imgtrain"+str(counter)+".jpg", result)
            counter+=1


    cv.imshow('frame', result)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

