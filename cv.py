import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
import math
from tf_classify import classify, coneify, forwardify



def convex_hull_pointing_up(ch):
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.9:
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center:
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False
        
    return True

def convex_hull_squared(ch):
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio > 0.7 and aspect_ratio < 1.3:
        vertical_center = y + h / 2


        return True
    else:
        return False

lower_yellow_e = np.array([20, 115, 115])
upper_yellow_e = np.array([35, 255, 255])

if not cap.isOpened():
    print("ERROR")
    exit()
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow_e, upper_yellow_e)
    result = cv.bitwise_and(frame, frame, mask = mask)

    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    copy_result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

    ret,result = cv.threshold(result,70,255,0)
    ret,copy_result = cv.threshold(copy_result,60,255,0)

    kernel = np.ones((5, 5))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    result = cv.medianBlur(result, 5)
    result = cv.medianBlur(result, 5)

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    (h, w) = result.shape[:2]
    for i in range(h):
        for j in range(w):
            if result[i][j]:
                result[i][j]=255

    result = cv.Canny(result, 60, 200)


    contours, _ = cv.findContours(np.array(result), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    approx_contours = []

    for c in contours:
        approx = cv.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    approx_contours = sorted(approx_contours, key=lambda c: cv.contourArea(c), reverse=True)
    try:
        approx_contours = [approx_contours[0]]
    except:
        pass

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv.convexHull(ac))


    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv.convexHull(ch))

    cones = []
    bounding_rects = []
    b2 = []
    for ch in convex_hulls_3to10:
        #if convex_hull_pointing_up(ch):
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_rects.append(rect)
        b2.append(cv.minAreaRect(ch))

    ret,copy_result = cv.threshold(copy_result,60,255,0)
    mask = cv.inRange(copy_result, (0, 0, 0), (255, 255, 255))
    copy_result = cv.bitwise_and(frame, copy_result, mask=mask)
    y_copy_result = copy_result

    if len(bounding_rects) >= 1:
        try:
            dims = bounding_rects[0]
            y_copy_result = copy_result[dims[1]-20:(dims[1]+dims[3])+20, (dims[0])-10:(dims[0]+dims[2])+20]
            cv.imshow("FRAME-K", y_copy_result)
        except:
            pass

    coneis = int(str(coneify(copy_result))[6])
    forwardis = int(str(forwardify(copy_result))[6])

    try:
        #coneis = int(str(coneify(y_copy_result))[6])
        pass
    except:
        pass

    #copy_result=cv.cvtColor(copy_result, cv.COLOR_GRAY2RGB)
    for rect in bounding_rects:
        if coneis==1:# or forwardis==1:
            copy_result = cv.rectangle(copy_result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
    
    if y_copy_result.shape[0]*y_copy_result.shape[1] < 2500:
        y_copy_result=copy_result

    #cv.drawContours(qresult, contours, -1, (0,255,0), 2)

    #ret,thresh = cv.threshold(copy_result,30,255,0)
    #contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    


    copy_result = cv.putText(copy_result, "ENABLED: "+str(coneis), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    #if counter%1==0:
    #        try:
    #            cv.imwrite("images/not_is_cone/cone" + str(counter+0) + ".jpg", y_copy_result)
    #        except:
    #            pass
    
    #counter +=1
    if len(cones) >= 1 and coneis==1:
        if convex_hull_pointing_up(cones[0]):
            copy_result = cv.putText(copy_result, "ORIENTATION: UP", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        elif convex_hull_squared(cones[0]):
            copy_result = cv.putText(copy_result, "ORIENTATION: SQUARE: "+str(forwardis), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        else:
            copy_result = cv.putText(copy_result, "ORIENTATION: SIDE", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    
    cv.imshow('FRAME', copy_result)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

