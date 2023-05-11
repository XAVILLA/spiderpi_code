#from HiwonderSDK import Mpu6050
from MPU6050.MPU6050 import MPU6050
import math
import numpy as np
from matplotlib import pyplot as plt
from utils import DELTA_T
import threading

# First import the library
import pyrealsense2 as rs

class Observer:
    def __init__(self):
        
        
        self.a = [0, 0, 0]
        self.rpy = [0,0,0]
        self.g = [0, 0, 0]
    
        self.last_acceleration = None
        self.last_angular_velocity = None

        self.history = []
        self.pipe = rs.pipeline()

        # Build config object and request pose data
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)
        self.no_frame = False

        # Start streaming with requested config
        print("pipeline starting")
        self.pipe.start(self.cfg)
        print("pipeline started")

    def update_mpu_obs(self):
        return 0
    def get_observation(self):
#         return np.array([0, 0])#
#         try:
#             acceleration = self.mpu.get_accel_data(g=False) # m / s^2
#             self.last_acceleration = acceleration
#         except Exception as e:
#             print(e)
#             if self.last_acceleration is not None:
#                 acceleration = self.last_acceleration
#             else:
#                 acceleration = {"x": 0, "y": 0, "z": 0}
# 
#         try:
#             angular_velocity = self.mpu.get_gyro_data() # degrees per second
#             self.last_angular_velocity = angular_velocity
#         except Exception as e:
#             print(e)
#             if self.last_angular_velocity is not None:
#                 angular_velocity = self.last_angular_velocity
#             else:
#                 angular_velocity = {"x": 0, "y": 0, "z": 0}
# 
#         ax = acceleration["x"]
#         ay = acceleration["y"]
#         az = acceleration["z"]
#         roll = math.atan2(ax, az) # radians
#         pitch = math.atan2(ay, az) # radians
#         
#         roll = math.atan2(ay , az)
#         pitch = math.atan2(-ax , math.sqrt(ay * ay + az * az))
        
        # print(ax, ay, az, roll, pitch)
        #roll = pitch = 0
        #ax = ay = 0
        # az = -9
#         observation = np.array([
#             self.a[0],#self.ax,
#             self.a[1],#self.ay,
#             self.a[2],#self.az,
#             self.rpy[0], #self.rpy[0],#self.roll,#roll,
#             self.rpy[1], #self.rpy[1],#self,pitch,#pitch,
#             self.g[0],#self.roll,
#             self.g[1],#self.gx,
#             self.g[2],#self.gy,
#             #self.gz
#             #np.radians(angular_velocity["x"]), # DMP doesn't give angular velocity, only attitude, acceleration, gravity. 
#             #np.radians(angular_velocity["y"]),
#             # np.radians(angular_velocity["z"]),
#         ])
#         
#         #if len(self.history) > 0:
#             #alpha = 0.75
#             #observation = observation*(1-alpha) + self.history[-1]*alpha
#             # observation[3:5] = observation[3:5]*(1-alpha) + self.history[-1][3:5]*alpha
#         self.history.append(observation)
        # if self.no_frame:
        #     pose_np = np.array([0, 0, 0])
        #     return pose_np
   
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=50)
        except RuntimeError as e:
            print(e)
            # self.no_frame = True
            pose_np = np.array([0, 0, 0])
            return pose_np

        # Fetch pose frame
        pose = frames.get_pose_frame()
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
            #print(yaw)
            if math.isnan(roll):
                roll = 0
            if math.isnan(pitch):
                pitch = 0
            if math.isnan(yaw):
                yaw = 0
        print(roll, pitch, yaw)
        # return np.array([0, 0, 0])
        return np.array([roll,-pitch, -yaw])

            
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

        