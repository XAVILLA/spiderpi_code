import time
import numpy as np
import math
import cv2
from observer import Observer
from high_level_observer import HighLevelObserver
from matplotlib import pyplot as plt
from utils import *


def hl_features(hl_observation):
    img_copy = hl_observation.copy()
    img_h, img_w = hl_observation.shape[:2]

    blob = cv2.dnn.blobFromImage(img_copy, 1, (250, 250), [104, 117, 123], False, False)
    
    #net.setInput(blob)
    #detections = net.forward() # Calculated recognition
    
    return img


def main():
    base_policy_onnx_path = "/home/pi/Desktop/flat_spiderpi_base_policy.onnx"
    observation_encoder_onnx_path = "/home/pi/Desktop/flat_spiderpi_obs_encoder.onnx"
    base_policy = cv2.dnn.readNetFromONNX(base_policy_onnx_path)
    observation_encoder = cv2.dnn.readNetFromONNX(observation_encoder_onnx_path)
    observer = Observer()
    
    use_prior = False
    # hl_observer = HighLevelObserver()

    prev_pred = np.zeros(18)
    f_c = 1 # 3 or 4 Hz
    rc = 1/(2*np.pi*f_c)
    alpha = DELTA_T / (rc + DELTA_T)

    transition_length = ROLLOUT_STEPS / 10
    v_max = 0.4
    desired_vels = []

    for i in range(ROLLOUT_STEPS):
        observation = observer.get_observation()
        # hl_observation = hl_observer.get_observation()

        if i < transition_length:
            v_t = (i / transition_length) * v_max
        elif i < (ROLLOUT_STEPS - transition_length):
            v_t = v_max
        else:
            v_t = (1 - ((i - (ROLLOUT_STEPS - transition_length)) / transition_length)) * v_max
        desired_vels.append(v_t)
        command = [
            v_t,
            0,
            0
        ]

        policy_input = np.concatenate((
            observation,
            prev_pred,
            command
        ))
        observation_encoder.setInput(policy_input[None])
        z_t = policy.forward()
        prediction = base_policy.setInput(z_t)[0]
        prediction = np.clip(prediction, a_min=-1.0, a_max=1.0)
        prediction = prediction*alpha + prev_pred*(1-alpha)
        prev_pred = prediction

        joint_angles = policy_outputs_to_angles(sim_direction_to_real_direction(prediction))
        set_joint_angles(joint_angles)
    linear_velocities = observer.get_linear_velocity(DELTA_T)
    ts = list(range(len(linear_velocities)))
    plt.plot(ts, linear_velocities)
    plt.plot(ts[:-1], desired_vels)
    plt.show()
    
    accelerations = observer.ax_buffer
    estimated_accelerations = [(desired_vels[i] - desired_vels[i-1]) / DELTA_T for i in range(1, len(desired_vels))]
    plt.plot(ts[1:], accelerations)
    plt.plot(ts[2:], estimated_accelerations)
    plt.show()


if __name__ == "__main__":
    reset_joints()
    #main()
    #reset_joints()    

