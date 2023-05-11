import math
import numpy as np
from matplotlib import pyplot as plt
from utils import DELTA_T
import threading
# f1230127 realsense l515
# 2322110220 t265
# First import the library
import pyrealsense2 as rs
import cv2

class test:
    def __init__(self):
        self.pipe2 = rs.pipeline()
        self.cfg2 = rs.config()
        self.cfg2.enable_device('f1230127')
        self.cfg2.enable_stream(rs.stream.depth, 320, 240)
        profile = self.pipe2.start(self.cfg2)
        print(profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().ppx)
        print("pipeline started")
        
    def get_observation(self):
        frames2 = self.pipe2.wait_for_frames()
        depth_frame = frames2.get_depth_frame()
        print(depth_frame.profile.as_video_stream_profile().intrinsics.ppx)
        depth_image = np.asanyarray(depth_frame.get_data())
        print(depth_image.shape)
        depth_image_convert = cv2.convertScaleAbs(depth_image, alpha=0.25)
        #depth_image_convert = depth_image
        depth_image_resized = cv2.resize(depth_image, (160, 160), interpolation = cv2.INTER_NEAREST)
        #print(np.min(depth_image_convert), np.max(depth_image_convert))
        depth_image_thres = np.zeros(depth_image.shape)
        depth_image_thres[depth_image_convert == 0] = 1
        depth_colormap = cv2.applyColorMap(depth_image_convert, cv2.COLORMAP_JET)
        #images = np.hstack((color_image, depth_colormap))
        images = depth_image
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        
        
        
if __name__ == "__main__":
    t = test()
    for i in range(1000):
        t.get_observation()
        
    