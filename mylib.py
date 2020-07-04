# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 20:26:39 2020

@author: madhup
"""

import numpy as np

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
