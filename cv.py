import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
import math

def convex_hull_pointing_up(ch):
    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.8:
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

lower_yellow_e = np.array([18, 95, 95])
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
    ret,copy_result = cv.threshold(copy_result,70,255,0)

    kernel = np.ones((5, 5))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
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
    for ch in convex_hulls_3to10:
        #if convex_hull_pointing_up(ch):
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_rects.append(rect)

    

    for rect in bounding_rects:
        copy_result = cv.rectangle(copy_result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)

    #cv.drawContours(qresult, contours, -1, (0,255,0), 2)

    #ret,thresh = cv.threshold(copy_result,30,255,0)
    #contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    copy_result = cv.putText(copy_result, "ENABLED: 0", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    if len(cones) >= 1:
        counter+=1
        if convex_hull_pointing_up(cones[0]):
            copy_result = cv.putText(copy_result, "ORIENTATION: UP", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        elif convex_hull_squared(cones[0]):
            if counter%6==0:
                cv.imwrite("images/c_one"+str(counter)+".jpg", result)
            copy_result = cv.putText(copy_result, "ORIENTATION: SQUARE", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        else:
            copy_result = cv.putText(copy_result, "ORIENTATION: SIDE", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    
    cv.imshow('frame', copy_result)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

