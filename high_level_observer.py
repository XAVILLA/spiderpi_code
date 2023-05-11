#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import math
import time
import threading
import numpy as np

from HiwonderSDK import LABConfig
from HiwonderSDK import Camera
from HiwonderSDK import import_path
import HiwonderSDK.Board as Board
import HiwonderSDK.Sonar as Sonar
from HiwonderSDK.CameraCalibration.CalibrationConfig import *


CAMERA_PULSE_MAX = 2500
CAMERA_PULSE_MIN = 500
CAMERA_PULSE_RANGE = CAMERA_PULSE_MAX - CAMERA_PULSE_MIN

CAMERA_PULSE_STALL_MAX = 0.7 * CAMERA_PULSE_MAX
CAMERA_PULSE_STALL_MIN = CAMERA_PULSE_RANGE/2 - 0.7 * (CAMERA_PULSE_RANGE/2 - CAMERA_PULSE_MIN)

CAMERA_ANGLE_MAX = np.pi / 2
CAMERA_ANGLE_MIN = -np.pi / 2
CAMERA_ANGLE_RANGE = CAMERA_ANGLE_MAX - CAMERA_ANGLE_MIN


class HighLevelObserver:
    def __init__(self):
        # run thread for passive movement in bakcground (Swing)
        #th = threading.Thread(target=move)
        #th.setDaemon(True)
        #th.start()

        # Loading parameters
        param_data = np.load(calibration_param_path + '.npz')
        mtx = param_data['mtx_array']
        dist = param_data['dist_array']
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)    
        self.HWSONAR = Sonar.Sonar()
        self.camera = Camera.Camera()
        self.camera.camera_open()
        self.reset()
        
    
    def get_observation(self):
        while True:
            img = self.camera.frame
            if img is not None:
                frame = img.copy()
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)  # Distortion correction    
                break
            else:
                time.sleep(0.01)
        return frame
        

    # Initial position
    def reset(self):
        self.HWSONAR.setRGBMode(0)
        self.HWSONAR.setRGB(1, (0, 0, 0))
        self.HWSONAR.setRGB(2, (0, 0, 0))     
        self.set_camera_joint_angles(0, 0)
        
    
    def close(self):
        self.camera_close()
        cv2.destroyAllWindows()
        

    #######################################
    #        Movement Utilities           #
    #######################################

    def camera_joint_angles_to_pulse(self, angle):
        angle_scaled = (angle - CAMERA_ANGLE_MIN) / CAMERA_ANGLE_RANGE
        return angle_scaled * CAMERA_PULSE_RANGE + CAMERA_PULSE_MIN
        
                
    def set_camera_joint_angles(self, pitch, yaw, duration=500):
        pitch_pulse = self.camera_joint_angles_to_pulse(pitch)
        yaw_pulse = self.camera_joint_angles_to_pulse(yaw)
        
        pitch_pulse = max(min(CAMERA_PULSE_STALL_MAX, pitch_pulse), CAMERA_PULSE_STALL_MIN)
        yaw_pulse = max(min(CAMERA_PULSE_MAX, yaw_pulse), CAMERA_PULSE_MIN)
        
        Board.setPWMServoPulse(1, pitch_pulse, duration)
        Board.setPWMServoPulse(2, yaw_pulse, duration)  
    
