#! /usr/bin/env python

#from numba import jit, cuda
import numpy as np
import cv2 as cv
import math
#from tf_classify import coneify, forwardify
import time
import threading
#from keras.models import load_model
from networktables import NetworkTables

NetworkTables.initialize(server='roborio-846-frc.local')
table = NetworkTables.getTable("GamePieces")

print("SCRIPT START")
print("SCRIPT: imported all necessary libs")

# fwd= open('fwd.txt', 'r')
total=0
frames=0
np.set_printoptions(suppress=True)

#@jit(target_backend='cuda')
def convex_hull_pointing_up(ch, b):
    points_above_center, points_below_center = [], []

    (zx, zy), (width, height), angle = b
    
    x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.9:

        # print('afafafaewiofj')
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
                return 0
    else:
        return 0
        
    return 1
    
def convex_hull_side(ch, b):
    points_above_center, points_below_center = [], []
    
    x, y, width, height = b

    #x, y, w, h = cv.boundingRect(ch)
    aspect_ratio = width / height

    if aspect_ratio>1.1:
        return 1
    else:
        return 0



def run_cube(frame):
    k_cube=1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #hsv = cv.medianBlur(hsv, 5)
    
    lower = np.array([114, 30, 60])
    upper = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower, upper)
    
    result = cv.bitwise_and(frame, frame, mask = mask)

    ret,result = cv.threshold(result,1,255,0)

    copy_result = result

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    #result = cv.medianBlur(result, 5)

    np_result = np.array(result)
    np_result[np_result != 0] = 255
    result = cv.merge((np_result, np_result, np_result))

    #cv.GaussianBlur(result, (51, 51), 0)
    kernel = np.ones((3, 3))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    #result = cv.medianBlur(result, 5)
    #result = cv.medianBlur(result, 5)

    result = cv.Canny(result, 60, 120, -1)

    kernel = np.ones((1, 1))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(np.array(result), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    approx_contours = []

    for c in contours:
        approx = cv.approxPolyDP(c, 8, closed = True)
        approx_contours.append(approx)

    sorted_contours= sorted(contours, key=cv.contourArea, reverse= True)
    
    if approx_contours:
        approx_contours = [approx_contours[0]]

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
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_rects.append(rect)
        b2.append(cv.minAreaRect(ch))

    if b2:
        (zx, zy), (width, height), angle  = b2[0]
        if width/height >= 0.5 and width/height <= 2 and width*height>=100:
            print("CUBE DETECTION: " + str(width) + "," + str(height) + " | " + str(zx) + "," + str(zy))
            pass


def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    
    return result

def deg2rad(n):
    return 3.14159*n/180

def dist(point, hc):
    # hc=3.875
    hm=28
    y=864
    x = 1056
    vtheta=deg2rad(33)
    htheta=deg2rad(47.92)
    am=deg2rad(56.66)
    total=[0, 0, 0]

    rat = vtheta/y
    # x = point[1]
    # y = point[0]

    # nx = (2 / (x_res)) * (x - x_res - 0.5)
    # ny = (2 / y_res) * (y_res - 0.5 - y)

    # vpw = 2 * 0.452379
    # vph = 2 * 0.2067735

    # x = (vpw / 2) * nx
    # y = (vph / 2) * ny

    # ax = 180*(math.atan(x))/3.14159
    # ay = 180*math.atan(y)/3.14159
    # print(ay+mount_angle)


    # dist = (mount_height - hc) / (np.tan(np.deg2rad(ay + mount_angle)))

    # total[0] = ax
    # total[1] = ay
    # total[2] = dist

    # return total

    #Second method
    #dist = 


    # print(point[0])
    ny = y/2-point[0]
    # print(ny)
    nx = x/2-point[1]
    # print(nx)

    ay= math.atan(math.tan(vtheta/2)*ny/(y/2))
    print(ay*180/3.14159)
    print((ay+am)*180/3.14159)
    print(hm-hc)
    posy=math.tan(am+ay)*(hm-hc) 
    # print(posy)
    tx=math.atan(nx*math.tan(htheta/2)/(x/2))
    # print(alpha)
    # print(tx)
    print(posy/math.cos(tx))
    posx=posy*math.tan(tx)

    total[0]=posx
    total[1]=posy
    total[2]=hc
    return total

def run_cone(frame):
    global total
    global frames
    # try:
    k_cone=1
    k_Fwd=-0.05
    frame=cv.resize(frame, (frame.shape[1]*3, frame.shape[0]*3), interpolation=cv.INTER_AREA)


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #mask = cv.inRange(hsv, lower_yellow_e, upper_yellow_e)
    #result = cv.bitwise_and(frame, frame, mask = mask)

#---
    upper_orange1 = np.array([0, 227, 227])
    upper_orange2 = np.array([34, 255, 255])

    imgThreshHigh = cv.inRange(hsv, upper_orange1, upper_orange2)

   # mask = cv.bitwise_and(imgThreshHL, imgThreshHigh)

    result = cv.bitwise_and(frame, frame, mask = imgThreshHigh)

    #result = cv.bitwise_or(imgThreshLow, imgThreshHigh)
#--

    kernel = np.ones((5,5),np.uint8)

    result = cv.erode(result, kernel, iterations = 3)
    result = cv.dilate(result, kernel, iterations = 2)


    # result = zoom_at(result, 3, coord=(77, 97))

    e_result=result

    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    #copy_result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    #copy_result = cv.cvtColor(copy_result, cv.COLOR_BGR2GRAY)

    ret,result = cv.threshold(result,70,255,0)
    #ret,copy_result = cv.threshold(copy_result,60,255,0)

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)



    result = cv.medianBlur(result, 5)

    kernel = np.ones((1, 1))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)


    result = cv.medianBlur(result, 5)
    # result = cv.medianBlur(result, 5)
    #result = cv.blur(result, (5, 5))


    np_result = np.array(result)
    np_result[np_result != 0] = 255
    #for i in range(h):
    #    for j in range(w):
    #        if result[i][j]:
    #            result[i][j]=255
    

    result = cv.merge((np_result, np_result, np_result))
    fake_e_result = result

    result = cv.Canny(result, 60, 200)


    contours, _ = cv.findContours(np.array(result), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    approx_contours = []

    for c in contours:
        approx = cv.approxPolyDP(c, 0.06 * cv.arcLength(c, True), closed = True)
        approx_contours.append(approx)

    approx_contours = sorted(approx_contours, key=lambda c: cv.contourArea(c), reverse=True)
    approx_contours = [approx_contours[0]]



    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv.convexHull(ac))


    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 6:
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

    dims = bounding_rects[0]
    y_copy_result = fake_e_result[dims[1]:(dims[1]+dims[3]), (dims[0]):(dims[0]+dims[2])]

    cv.rectangle(fake_e_result, (bounding_rects[0][0], bounding_rects[0][1]), (bounding_rects[0][0]+bounding_rects[0][2], bounding_rects[0][1]+bounding_rects[0][3]), (255, 255, 255), 1)

    box=cv.boxPoints(b2[0])

    boxCenter=((box[0][0]+box[2][0])/2, (box[0][1]+box[2][1])/2)

    mask = y_copy_result[:, :, 2] != 0


    mean_position = np.mean(np.argwhere(mask), axis=0)
    mean_position=(mean_position[0]+dims[1], mean_position[1]+dims[0])


    verticalWeightage= (mean_position[0]-boxCenter[1])/b2[0][1][1]
    horizontalWeightage= (mean_position[1]-boxCenter[0])/b2[0][1][0]

    fake_e_result = cv.circle(fake_e_result, (int(boxCenter[0]), int(boxCenter[1])), 3, (244, 0, 0), 3)
    fake_e_result = cv.circle(fake_e_result, (int(mean_position[1]), int(mean_position[0])), 3, (0, 244, 0), 3)
    fake_e_result=cv.circle(fake_e_result, (528, 432), 30, (0, 0, 255), 30)
    fake_e_result=cv.drawContours(fake_e_result, [np.int0(cv.boxPoints(b2[0]))], 0, (0, 0, 255), 2)
    print(str(len(np.array(result))), "   ", str(len(np.array(result)[0])))
    # print(boxCenter)

    cv.imwrite('e_result.png', fake_e_result)
    cv.imshow('e_result.png', fake_e_result)





    if b2:# and coneis==1:
        (zx, zy), (width, height), angle  = b2[0]
        if width/height >= 0.2 and width/height <= 5 and width*height>=25:
            # print(bounding_rects[0][2]/bounding_rects[0][3])
            base_length=0 
            orientation=0
            if verticalWeightage>0.05 and not(convex_hull_side(cones[0], bounding_rects[0])):
                orientation=0
            elif not(verticalWeightage>0.05) and convex_hull_side(cones[0], bounding_rects[0]):
                orientation=1
            elif verticalWeightage<k_Fwd:  
                orientation=2
            else:
                orientation=3
            if orientation==1:
                base_length=height
            else:
                base_length=width
            print("CONE DETECTION: ", str(dist(mean_position, 4)))

            total+=verticalWeightage
            frames+=1
            distance = base_length*k_cone
            angle=52.29*(122-zx)/122            
            table.putNumber("coneDistance", distance)
            table.putNumber("coneAngle", angle)
            table.putNumber("coneOrientation", orientation)
            return
    table.putNumber("coneDistance", 0)
    table.putNumber("coneAngle", 0)
    table.putNumber("coneProbability", 0)
    # print('xcfghjkgfdfghjkl', cones[0])
    cv.drawContours(result, cones, 0, (255, 255, 255), 3)
    cv.imwrite('awf.png', result)
    # except Exception as e:
    #     print(e)
    


#cuda R function
#@jit(target_backend='cuda')
def CUDA_OPTIMIZED_RUN846():
    print("SCRIPT: ATTEMPTING CAM 1")
    cap = cv.VideoCapture(1,cv.CAP_DSHOW)#"/home/tech/gamepiece.com/Cone.mov")
    if cap.isOpened()==False:
        print("SCRIPT: ATTEMPTING CAM 2")
        cap = cv.VideoCapture(0)
    if cap.isOpened()==False:
        print("SCRIPT: CAM CONN FAILURE")
    else:
        print("SCRIPT: CAM CONN")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)

    counter = 0
    otime = time.time()
    if cap.isOpened():
        t1 = threading.Thread(target=run_cone, args=(cap,))
        t2 = threading.Thread(target=run_cube, args=(cap,))
        t1.start()
        t2.start()
        while(True):
            counter+=1
            # frame = cv.imread('cone.jpg')
            _, frame = cap.read()
            #---------CUBE-------
            t1 = threading.Thread(target=run_cone, args=(frame,))
            t2 = threading.Thread(target=run_cube, args=(frame,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            #---------CONE-------

            if counter % 10 == 0:
                counter = 0
                print("FPS: " + str(30/(time.time()-otime)))
                otime = time.time()
            
            


            #if cv.waitKey(1)== ord('q'):
            #    break

        
    cv.destroyAllWindows()
    cap.release()





CUDA_OPTIMIZED_RUN846()
