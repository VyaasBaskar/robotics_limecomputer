import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
import math
#from tf_classify import coneify, forwardify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time

np.set_printoptions(suppress=True)

model2 = load_model("cone_model/keras_Model.h5", compile=False)

class_names2 = open("cone_model/labels.txt", "r").readlines()

model3 = load_model("next_model2/keras_Model.h5", compile=False)

class_names3 = open("next_model2/labels.txt", "r").readlines()

def coneify(cvimg):
    #cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    image = cv.resize(cvimg, (224, 224), interpolation=cv.INTER_AREA)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model2.predict(image)
    index = np.argmax(prediction)
    class_name = class_names2[index]
    confidence_score = prediction[0][index]

    cn = class_name[2:]

    #if confidence_score <= 0.98 and cn[6] == "1":
    #    cn = "Class 2"
    #    print("E: CHANGE")

    return cn

def forwardify(cvimg):
    cvimg = cv.cvtColor(cvimg, cv.COLOR_BGR2RGB)
    image = cv.resize(cvimg, (224, 224), interpolation=cv.INTER_AREA)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model3.predict(image)
    index = np.argmax(prediction)
    class_name = class_names3[index]
    confidence_score = prediction[0][index]

    #print(confidence_score)

    cn = class_name[2:]

    #if cn[6] == "2" and confidence_score < 1.0:
    #     cn = "Class 1"
    return cn


def convex_hull_pointing_up(ch, b):
    points_above_center, points_below_center = [], []

    (zx, zy), (width, height), angle = b
    
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
    

def convex_hull_squared(ch, b):
    points_above_center, points_below_center = [], []
    
    (x, y), (width, height), angle = b

    #x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = width / height

    if aspect_ratio > 0.65 and aspect_ratio < 1.35:
        return True
    else:
        return False

lower_yellow_e = np.array([18, 95, 95])
upper_yellow_e = np.array([37, 255, 255])

if not cap.isOpened():
    print("ERROR")
    exit()
counter = 0
while True:
    counter +=1
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
    #result = cv.blur(result, (5, 5))

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    np_result = np.array(result)
    np_result[np_result != 0] = 255
    #for i in range(h):
    #    for j in range(w):
    #        if result[i][j]:
    #            result[i][j]=255

    result = cv.merge((np_result, np_result, np_result))

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
        b2.append(cv.minAreaRect(ch))#cv.boxPoints(cv.minAreaRect(ch)))

    ret,copy_result = cv.threshold(copy_result,60,255,0)
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
        coneis = int(str(coneify(y_copy_result))[6])
        forwardis = int(str(forwardify(y_copy_result))[6])
        pass
    except:
        pass

    #copy_result=cv.cvtColor(copy_result, cv.COLOR_GRAY2RGB)
    """
    if coneis==1 and bounding_rects:
        rect = bounding_rects[0]
        try:
            copy_result = cv.drawContours(copy_result,[np.int0(cv.boxPoints(b2[0]))],0,(0,0,255),2)
        except:
            copy_result = cv.rectangle(copy_result, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
    
    if y_copy_result.shape[0]*y_copy_result.shape[1] < 2500:
        y_copy_result=copy_result

    #cv.drawContours(qresult, contours, -1, (0,255,0), 2)

    #ret,thresh = cv.threshold(copy_result,30,255,0)
    #contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    


    copy_result = cv.putText(copy_result, "ENABLED: "+str(coneis), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    #if counter%1==0:
    #        try:
    #            cv.imwrite("images/is_forward_2/cone" + str(counter+0) + ".jpg", y_copy_result)
    #        except:
    #            pass

    
    
    #counter +=1
    if len(cones) >= 1 and coneis==1:
        (x, y), (width, height), angle = b2[0]
        if convex_hull_pointing_up(cones[0], b2[0]):
            copy_result = cv.putText(copy_result, "ORIENTATION: UP", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        elif convex_hull_squared(cones[0], b2[0]):
            copy_result = cv.putText(copy_result, "ORIENTATION: SQUARE: "+str(forwardis), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        else:
            copy_result = cv.putText(copy_result, "ORIENTATION: SIDE", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    
    #cv.imshow('FRAME', copy_result)

    """
    if b2 and coneis==1:
        (zx, zy), (width, height), angle  = b2[0]
        if width/height >= 0.75 and width/height <= 1.25 and width*height>=2500:
            print("DETECTION: " + str(width) + "," + str(height) + " | " + str(zx) + "," + str(zy))
            pass

    if counter % 30 == 0:
        counter = 0
        print("FPS: " + str(30/(time.time()-otime)))
        otime = time.time()

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()

