# d435_segmentation.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression
import imutils

def detect_body_parts(frames_list):
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



def depth_in_valid_range(depth):
    return (depth >= min_valid_dep and depth <= max_valid_dep)

# This function is kept for extended processing ... need to be used.
def do_morphing(color_frame, depth_frame):
    
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)        
    
    gaussian_kernel = 3
    gray_blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
    #cv2.imshow("Gray with Thr", gray_blur)
    return (gray_blur)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (gray)

# A replacement for cv2.inRange( ). cv2 function works with uint8 values and
# When we convert actual depth to uint8, we do rounding which loses depth
# information to some extent
def get_depth_thresh_mask (dep_data):
    return (np.uint8(255 * np.logical_and(dep_data>=min_valid_dep, dep_data<=max_valid_dep)))

def detect_ped_with_hog(aligned_frames, frame_no):
    
    # For HOG, resizing the original frame to 400 sometimes helps. Set the
    # below variable to True if resizing is to be done.
    do_resize = False
    
    # Option to display the processed frames with bounding boxes
    # If set to False, no images will be displayed, just the processing
    # continues and final stats calculated in main program.
    display_images = True
    display_filtered_img = False
    display_mask_img = False
    # Extract the color and depth frames from the stream
    color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data())
    depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise in the image
    gaussian_kernel = 3
    gray_image = cv2.GaussianBlur(gray_image, (gaussian_kernel, gaussian_kernel), 0)
    if display_filtered_img:
        image = cv2.bitwise_and(color_frame, color_frame, mask=get_depth_thresh_mask(depth_frame))
    else:
        image = [0]
    
    # Resize the Image if desired
    width = color_frame.shape[1]
    if do_resize and width > 400:
        width = 400
        gray_image = imutils.resize(gray_image, width) 
        color_frame = imutils.resize(color_frame, width)
        if display_filtered_img:
            image = imutils.resize(image, width)

   
    # Detecting all the regions in the Image that has a pedestrians inside it
    # winStride, padding & scale parameters control the performance of HOG
    pad = 8
    (regions, _) = hog.detectMultiScale(gray_image,  
                                        winStride=(4, 4), 
                                        padding=(pad, pad), 
                                        scale=1.05) 
    
    # Whether a pedestrian is detected or not
    found = False
    
    # Get the rectangles from detected regions
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
    
    # Apply Non Maximum Suppression to reduce overlapping bounding boxes
    nms_boxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # There could be multiple pedestrians in the frame. Mark each pedestrian
    box = 1
    for (xA, yA, xB, yB) in nms_boxes:
        
        found = True
        # Default color for detected pedestrian bounding box. Will be changed
        # in below code if distance is within range
        col = (0,0,255)
        # Shrink the bounding box coordinates by "pad" size to get more closer
        # bounding box for the pedestrian
        xA += pad
        xB -= pad
        yA += pad
        yB -= pad
        
        # Use the bounding box region coordinates from NMS to retrieve depth 
        # data from depth frame
        # Try to find the depth of pedestrian using MEDIAN of depth values
        dep_data = depth_frame[yA:yB, xA:xB]
        
        # Find median value of all depth pixels, excluding invalid depths (value of 0)
        med_dep_act = np.median(dep_data[dep_data != 0])
        
        # Perform background subtraction - Remove all depth pixels which are
        # beyond Min & Max boundaries. Result is a pixel MASK that are
        # supposed to belong to the pedestrian
        dep_thr = get_depth_thresh_mask(dep_data)
        
        # Apply the maxk on depth pixel data
        valid_dep = cv2.bitwise_and(dep_data, dep_data, mask=dep_thr)
        
        # Find the median of the resultant pixels. This is our approximation
        # of pedestrian distance
        contour_depth = np.uint16(np.median(valid_dep[valid_dep !=0 ]))
        
        '''
        Below code can display two things as a stack
        1. The pedestrian bounding box region only cropped from original image
        2. The depth MASK as an image (gray scale)
        But doing so could slow down and reduce performance. hence putting
        under flag check
        '''
        if display_mask_img:
            hstack = np.hstack((color_frame[yA:yB, xA:xB], cv2.cvtColor(dep_thr, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Contour", hstack)
        
        # Below code is to handle occlusion. If a pedestrian is detected but
        # is occluded, I am using a threshold to determing whether to treat it
        # as proper detection, else, discard it as the pedestrian could be far
        # away and an object closer to camera could be occluding it.
        roi_size = dep_data.shape[0] * dep_data.shape[1]
        fg_pixels = valid_dep[valid_dep!=0]
        thr_roi_size = np.size(fg_pixels)
        
        #print(f"   Frame, Person# {frame_no, box} -- Filtered Med & Mean Depths -- \
        #      {contour_depth, mean_dep},  Med_Dep_Actual -- {med_dep_act}) , Allowed: {max_valid_dep*256}")
        
        # Check the ration of data that remains after background subtraction
        # Use the bounding box only if the baalance is > 25%
        # Threshold value is controllable
        if (thr_roi_size / roi_size) < 0.25:
            contour_depth = med_dep_act
        
        # Check if the calculated MEDIAN depth is in required range
        dep_in_rng =  depth_in_valid_range(contour_depth)
        
        if dep_in_rng:
            #print(f"InRange: Box Dim(X.Y): {xwidth, yB-yA} -- \
                #{contour_depth, med_dep}) Allowed: {max_valid_dep*256}, In_Range: {in_rng}")
            # Use a different color to indicate WITHIN range pedestrian
            col = (0,255,0)
        
        '''
        print(f'        ROI Shape: {dep_data.shape, roi_size}, \
              THR SHape: {valid_dep.shape, thr_roi_size}, Ratio: {thr_roi_size / roi_size}')
        print("")
        cv2.circle(color_frame, (cx, cy), 7, col, -1)

        '''
        
        cv2.rectangle(color_frame, (xA, yA), (xB, yB),  col, 2)
        if display_filtered_img:
            cv2.rectangle(image, (xA, yA), (xB, yB),  col, 2)
        
        if contour_depth <= 500:
            ped_dep = med_dep_act
        else:
            ped_dep = contour_depth
        
        ped_dep /= 1000
        dep_str = "Person: "+str(box)+" - "+str(ped_dep)+"m"
        
        # Display Frame# and Person count and Distance on the frame
        cv2.putText(color_frame, dep_str, (xA-10,yA),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        cv2.putText(color_frame, "Frame# "+str(frame_no), (10,10),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        
        if display_filtered_img:
            cv2.putText(image, dep_str, (xA-10, yA),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
            
            cv2.putText(image, "Frame# "+str(frame_no), (10,10),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))

        box += 1
    
    # Showing the output Image
    if display_filtered_img:
        both_img = np.hstack((color_frame, image))
        cv2.imshow("Image", both_img)
    elif display_images:
        cv2.imshow("Image", color_frame)
    
    return (color_frame, image, found)


if __name__ == "__main__":
    
    
    dirpath = os.getcwd()
    
    # Different video files that can be input to the program. Path is relatove
    # to current directory
    video_file = 'video\pmb_ped_video_daytime_multiple_ped.bag'
    #video_file = 'video\pmb_ped_video_multi_distance.bag'
    
    # Use this if we want to use absolute path
    #video_file=dirpath+"\\"+file_name

    # Check if the specified file exists
    print(video_file)    
    if not os.path.exists(video_file):
        print(f'Video file {video_file} does not exist .. check it. Exiting now')
        exit
    else:
        print(f'Video file {video_file} exists ... continuing to segmentation')
    
    config = rs.config()
    # This specifies that we are loading pre-recorded data 
    # rather than using a live camera.
    config.enable_device_from_file(video_file, repeat_playback = False)
    
    pipeline = rs.pipeline()  # Create a pipeline
    profile = pipeline.start(config)  # Start streaming
    # Saving to a .bag file is done before alignment, so we need to do it when
    # reading from .bag files too.
    align = rs.align(rs.stream.color)
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))
    
    # Set the minimum and maximum distances in millimeters(MM) in which we want
    # to detect pedestrians
    min_valid_dep = 500
    max_valid_dep = 5000

    # Below are haar cascade classifiers for detecting various body parts.
    # These are kept as optional extensions to the solution. If HoG is not
    # able to detect a pedestrians, we can run these additional classifiers
    # to see if we can detect parts of human body to decide if a pedestrian is
    # present
    extended_processing = False
    face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
    upperbody_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')
    lowerbody_cascade = cv2.CascadeClassifier('xml/haarcascade_lowerbody.xml')

    # Initialize the HOG feature descriptor along with its SVM classifier
    # for detecting pedestrians    
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Stores total identified pedestrian frames count
    ped_count = 0
    
    # Stores the total processed frames count
    frame_seq_no = 0
    
    while True: # Loop over the images until we decide to quit.
        # Get the next frameset.
        try:
            frame_seq_no += 1
            frames = pipeline.wait_for_frames()
        
            aligned_frames = align.process(frames)
            
            # Call the method to identify pedestrians. 3 values are returned
            # 1 - Original frame with bounding boxes added to detected pedestrians
            # 2 - Original frame with background subtraction + bounding boxes
            # 3 - Whether a pedestrian was detected in the input frame
            col_frm, filtered_frm, found = detect_ped_with_hog(aligned_frames, frame_seq_no)
            
            if (found):
                # Increament pedestrian count if detected  
                ped_count += 1
            elif extended_processing:                
                (res_image, thresh, found) = detect_body_parts(aligned_frames, True)
                comb_img = np.hstack((res_image, thresh))
                cv2.imshow('All Frames', comb_img)
                if (found):
                    ped_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except(RuntimeError):
            print("End of file reached... stopping now")
            break
        
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Released all devices. Processed_Frames, Ped  count: {frame_seq_no, ped_count}")
    print(f'Accuracy - {ped_count/frame_seq_no*100}')