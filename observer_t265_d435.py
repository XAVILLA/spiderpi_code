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

class Observer:
    def __init__(self):
        
        
        self.a = [0, 0, 0]
        self.rpy = [0,0,0]
        self.g = [0, 0, 0]
    
        self.last_acceleration = None
        self.last_angular_velocity = None

        self.depth_availble = True
        self.pose_availble = True

        self.history = []

        # Build config object and request pose data
        self.pipe1 = rs.pipeline()
        self.cfg1 = rs.config()
        self.cfg1.enable_device('2322110220')
        self.cfg1.enable_stream(rs.stream.pose)

        self.pipe2 = rs.pipeline()
        self.cfg2 = rs.config()
        self.cfg2.enable_device('f1230127')
        self.cfg2.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)

        # Start streaming with requested config
        print("pipeline starting")
        if self.pose_availble:
            self.pipe1.start(self.cfg1)
        if self.depth_availble:
            self.pipe2.start(self.cfg2)
        print("pipeline started")

    def update_mpu_obs(self):
        return 0

    def process_depth(self):

    def get_observation(self):
   
        frames = self.pipe1.wait_for_frames()

        # Fetch pose frame
        pose = frames.get_pose_frame()
        frames2 = self.pipe2.wait_for_frames()
        depth_frame = frames2.get_depth_frame()
        if depth_frame:
#             print("depth")
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_resized = cv2.resize(depth_image, (16, 16), interpolation = cv2.INTER_NEAREST)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #images = np.hstack((color_image, depth_colormap))
            images = depth_colormap
            
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
           
        if pose:
            # Print some of the pose data to the terminal
            data = pose.get_pose_data()

            # Euler angles from pose quaternion
            # See also https://github.com/IntelRealSense/librealsense/issues/5178#issuecomment-549795232
            # and https://github.com/IntelRealSense/librealsense/issues/5178#issuecomment-550217609

            w = data.rotation.w
            x = -data.rotation.z
            y = data.rotation.x
            z = -data.rotation.y

            pitch =  -math.asin(2.0 * (x*z - w*y)) #* 180.0 / math.pi;
            roll  =  math.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) #* 180.0/ math.pi;
            yaw   =  math.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) #* 180.0 / math.pi;
            print(yaw)
            if math.isnan(roll):
                roll = 0
            if math.isnan(pitch):
                pitch = 0
            if math.isnan(yaw):
                yaw = 0
        print(roll,pitch,yaw)
        obs = np.concatenate([np.array([roll,pitch]), depth_image_resized.flatten()])
        print(obs.shape)
        return obs

            
        #return np.concatenate((np.array([0, 0]), observation[np.array([5, 6, 3, 4])]))

    def get_euler_xyz_unwrapped(self, q):
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = q.w * q.w - q.x * \
            q.x - q.y * q.y + q.z * q.z
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(np.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = q.w * q.w + q.x * \
            q.x - q.y * q.y - q.z * q.z
        yaw = math.atan2(siny_cosp, cosy_cosp)
        

        return roll, pitch, yaw #roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

    def plot_history(self):
        history = np.array(self.history).T
        _, t = history.shape
        t = [i * DELTA_T for i in range(t)]
        fig, ax = plt.subplots(3, sharex=True)
        lin_acc = history[:3]
        ax[0].plot(t, lin_acc.T, label=["x", "y", "z"])
        ax[0].legend()
        ax[0].set_title("linear acceleration")
        ax[0].set_ylabel("g")
        lin_acc_mean, lin_acc_std = lin_acc.mean(), lin_acc.std()
        ax[0].set_ylim(lin_acc_mean-2*lin_acc_std, lin_acc_mean+2*lin_acc_std)
        gravity = history[5:]
        ax[1].plot(t, gravity.T, label=["x", "y", "z"])
        ax[1].legend()
        ax[1].set_title("projected gravity")
        ax[1].set_ylabel("g")
        gravity_mean, gravity_std = gravity.mean(), gravity.std()
        ax[1].set_ylim(gravity_mean-2*gravity_std, gravity_mean+2*gravity_std)
        attitude = history[3:5]
        ax[2].plot(t, attitude.T, label=["roll", "pitch"])
        ax[2].legend()
        ax[2].set_title("attitude")
        ax[2].set_ylabel("radians")
        ax[2].set_xlabel("seconds")
        attitude_mean, attitude_std = attitude.mean(), attitude.std()
        ax[2].set_ylim(attitude_mean-2*attitude_std, attitude_mean+2*attitude_std)
        plt.show()

    
    def get_linear_velocity(self, delta_t):
        velocities = [0]
        for ax in self.ax_buffer:
            velocity = velocities[-1] + delta_t*ax
            velocities.append(velocity)
        return velocities

        