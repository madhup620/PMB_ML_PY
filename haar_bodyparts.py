# -*- coding: utf-8 -*-

# d435_segmentation.py
import numpy as np
import cv2

def haar_detect_body_parts(frames_list):
    global min_dep, max_dep, nbrs
    
    body_parts = list(['u', 'l', 'f'])
    det_upper_body = False
    det_lower_body = False
    det_faces      = False
    color_frame = np.asanyarray(frames_list.get_color_frame().get_data())
    depth_frame = np.asanyarray(frames_list.get_depth_frame().get_data())
    
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    
    found = False
    scl = 1.05
    
    for part in body_parts:
        
        if found:
            break
        
        dist_str = "Dist: "
        if part == 'u':
            if (det_upper_body):
                bodies = upperbody_cascade.detectMultiScale(gray, scl, nbrs)
                dist_str = "Upper "+dist_str
            else:
                continue
        elif part == 'l':
            if (det_lower_body):
                bodies = lowerbody_cascade.detectMultiScale(gray, scl, nbrs)
                dist_str = "Lower "+dist_str
            else:
                continue
        elif part == 'f':
            if (det_faces):
                bodies = face_cascade.detectMultiScale(gray, scl, nbrs)
                dist_str = "Face "+dist_str
            else:
                continue
            
        col = (0,0,255)
        for x,y,wd,ht in bodies:
            cx = np.uint32(x + (wd/2))
            cy = np.uint32(y + (ht/2))
            contour_depth = depth_frame[cy, cx]
            if depth_in_valid_range(contour_depth):
                col = (0,255,0)
                found = True
            
            if found:
                if (contour_depth < min_dep):
                    min_dep = contour_depth
                
                if (contour_depth > max_dep):
                    max_dep = contour_depth
            cv2.circle(color_frame, (cx, cy), 10, col, -1)
            cv2.rectangle(color_frame, (x,y), (x+wd, y+ht), col, 2)
            cv2.putText(color_frame, dist_str+str(contour_depth),
                            (x-3, y), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0))
        
        #cv2.putText(color_frame, "Min: "+str(min_dep)+"  Max: "+str(max_dep), (10,10),
         #               cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0))
        
    
    #cv2.imshow('Color Img', color_frame)
    #cv2.imshow('Mask / Threshold', thresh)
    #thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return(color_frame, None, found)

# Set the minimum and maximum distances in millimeters(MM) in which we want
# to detect pedestrians
min_valid_dep = 500
max_valid_dep = 5000

def depth_in_valid_range(depth):
    return (depth >= min_valid_dep and depth <= max_valid_dep)

# A replacement for cv2.inRange( ). cv2 function works with uint8 values and
# When we convert actual depth to uint8, we do rounding which loses depth
# information to some extent
def get_depth_thresh_mask (dep_data):
    return (np.uint8(255 * np.logical_and(dep_data>=min_valid_dep, dep_data<=max_valid_dep)))

# Init the haar classifiers for face, lowerbody & upperbody parts
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')
