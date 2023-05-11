from HiwonderSDK import Board
import time
import numpy as np
from matplotlib import pyplot as plt

DELTA_T = 0.005 * 8
DELTA_T_MS = int(DELTA_T * 1000)
ROLLOUT_S = 60 #20 #10
ROLLOUT_STEPS = int(ROLLOUT_S / DELTA_T)
NUM_JOINTS = 18
TO_INVERT = [
    False, True, True, 
    False, True, True, 
    False, True, True, 
    True, False, False, 
    True, False, False, 
    True, False, False
]
INVERSION_ARRAY = -2*np.array(TO_INVERT, dtype=int) + 1


def policy_outputs_to_angles(outputs):
    angles = np.degrees(outputs) # convert from radians to degrees
    pose = "new"
    if pose == "old":
        return angles + 120#np.array([120,160,140,  120,160,140,   120,160,140,   120,80,100,   120,80,100, 120,80,100])
    elif pose == "new":
        # return angles + np.array([120, 100,  70, 120, 100,  70, 120, 100,  70, 120, 140, 170, 120, 140, 170, 120, 140, 170])
        # return angles + np.array([120, 105,  75, 120, 105,  75, 120, 105,  75, 120, 135, 165, 120, 135, 165, 120, 135, 165])
        return angles + np.array([120, 105,  65, 120, 105,  65, 120, 105,  65, 120, 145, 165, 120, 145, 165, 120, 145, 165])


def sim_direction_to_real_direction(radians):
    return INVERSION_ARRAY * radians    


def angle_to_pulse(angle):
    '''Returns the bus servo pulse corresponding to an angle.
       Angle has range [0, 240] corresponding to pulse range
       [0, 1000].
    '''
    angle = max(0, min(angle, 240))
    return int((angle / 240) * 1000)

def pulse_to_angle(pulse):
    '''Returns the bus servo pulse corresponding to an angle.
       Angle has range [0, 240] corresponding to pulse range
       [0, 1000].
    '''
    pulse = max(0, min(pulse, 1000))
    return int((pulse / 1000) * 240)

def set_joint_angles(joint_angles, duration=DELTA_T_MS):
    for idx, angle in enumerate(joint_angles):
        pulse = angle_to_pulse(angle)
        Board.setBusServoPulse(idx + 1, pulse, duration)
    #time.sleep(duration / 1000)


def reset_joints():
    home_angles = np.array([120]*18)
    home_angles_high = np.array([120,160,140,  120,160,140,   120,160,140,   120,80,100,   120,80,100, 120,80,100])
    home_angles_mid = (home_angles + home_angles_high)/2
    home_angles_test = np.array([120]*18)
    stable_displacement = np.array([0, -15, -47.5])
    #stable_displacement = np.array([0, 20, -10])
    for i in range(6):
        if i < 3:
            sign = 1
        else:
            sign = -1
        home_angles_test[i*3] += sign * stable_displacement[0]
        home_angles_test[i*3+1] += sign * stable_displacement[1]
        home_angles_test[i*3+2] += sign * stable_displacement[2]
    
    hiwonder_pose = np.array([120, 105,  75, 120, 105,  75, 120, 105,  75, 120, 135, 165, 120, 135, 165, 120, 135, 165])
    
    set_joint_angles(hiwonder_pose, duration=2000)

def read_joint_angles():
    for i in range(18):
        pulse = Board.getBusServoPulse(i+1)
        angle = pulse_to_angle(pulse)
        print(str(i)+"th joint:"+str(angle))
    
    
def plot_actions_history(actions_history, num_cols=3, num_rows=6):
    '''Takes a history of actions in the form of a list with length
       num_timesteps where each entry is a list with length num_actions.
    '''
    actions_history = np.array(actions_history).T
    num_actions, num_timesteps = actions_history.shape
    timesteps = list(range(num_timesteps))
    
    fig, axes = plt.subplots(num_rows, num_cols)
    for r in range(num_rows):
        for c in range(num_cols):
            i = r*num_cols + c
            axes[r, c].plot(timesteps, actions_history[i])
            axes[r, c].set_title(f"action {i+1}")
    plt.show()
