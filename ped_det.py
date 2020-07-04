# d435_segmentation.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys


if __name__ == "__main__":
    
    arg_cnt = len(sys.argv)
    if (arg_cnt < 2):
        print("No input file specified. Using default file\n") 
        # Different video files that can be input to the program. Path is relatove
        # to current directory
        video_file = 'video\pmb_ped_video_daytime_multiple_ped.bag'
        #video_file = 'video\pmb_ped_video_multi_distance.bag'
    else:
        video_file = "video"+"\\"+sys.argv[1]

    # Use this if we want to use absolute path
    #video_file=dirpath+"\\"+file_name

    # Check if the specified file exists
    print(f'Video file is: {video_file}\n')    
    dirpath = os.getcwd()
    if not os.path.exists(video_file):
        print(f'Video file {video_file} does not exist .. check it. Exiting now')
        exit
    else:
        print(f'Video file {video_file} exists ... continuing to segmentation')
    
    exit 
    
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
