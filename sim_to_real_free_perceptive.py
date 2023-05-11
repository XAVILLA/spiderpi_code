import time
import numpy as np
import math
import torch
from observer_perceptive import Observer_perceptive
from matplotlib import pyplot as plt
from utils import *

from actor_critic_recurrent_cnn_s2r import ActorCriticRecurrentCNNs2r


def main():
    policy_path = "/home/pi/Desktop/hexapod-isaac/sim_to_real/Jason's models/"
    policy_path += "rough_perceptive_maxrange1m_legmask_invalidmaskcorrect_depthinvalidmask2.3.pt"
    # policy_path += "rough_perceptive_maxrange1m_legmask/rough_perceptive_maxrange1m_legmask.pt"
    # policy_path += "no_legmask.pt"
    # policy_path += "potential best perceptive depth range 2.9.pt"
    #policy_path += "rough_spiderpi_Dec13_16-32-13_blind_withmount_patternedjoist_phase_two_yaw.pt"
    checkpoint = torch.load(policy_path, map_location=torch.device("cpu"))
    depth_shape = (32, 24)
    num_obs = 3+18+depth_shape[0]*depth_shape[1]
    num_actions = 18
    max_action = 2.094
#     action_scale = 1/64 #0.25
    action_scale = 1/3.6
    actor_critic = ActorCriticRecurrentCNNs2r(
        num_obs,
        num_obs,
        num_actions,
        actor_hidden_dims=[128, 64, 32], #[512, 256, 128],
        rnn_type='lstm',
        rnn_hidden_size=64, #512,
        rnn_num_layers=1
    )
    observation_encoder = actor_critic.memory_a
    base_policy = actor_critic.actor
    observation_encoder.load_state_dict(checkpoint["observation_encoder"])
    base_policy.load_state_dict(checkpoint["base_policy"])
    observation_encoder.eval()
    base_policy.eval()
    observer = Observer_perceptive()

    prev_actions = torch.zeros(num_actions)
    actions_history = []
    time_history = []
    pre_time = time.time()
    print("------------------------\n**  BEGINNING ROLLOUT **\n------------------------\n")
    last_inference_time = time.time()
    i = 0
    phi = 0
    delta_phi_leg = np.array([1,0,1,0,1,0]) * np.pi
    while i < ROLLOUT_STEPS:
        if time.time() - last_inference_time > DELTA_T:
            observation = torch.Tensor(observer.get_observation())
            # print(observation.shape)
            pose, perception = observation.split((3,32*24), dim=0)
            observation = [pose]
            observation.append(prev_actions)
            observation.append(perception)

            observation = torch.cat(observation, dim=-1).unsqueeze(0)
            z_t = observation_encoder(observation)
            actions = base_policy(z_t)
            actions = torch.clip(actions, min=-max_action / action_scale, max=max_action / action_scale)[0, 0]
            prev_actions = actions.clone() # save unscaled actions as prev actions as in sim
            actions = actions.detach().numpy()
            # stuff that only has to happen on the robot
            actions[9:12] = prev_actions[15:18].detach().numpy() # reorder some legs from sim order to real order
            actions[15:18] = prev_actions[9:12].detach().numpy()
            actions *= action_scale # scale down
            # actions *= np.array([1, 0.73, 0.73] * 6) # scale down joint 1 on all legs to avoid self collision
            actions_history.append(actions)
            
            joint_angles = policy_outputs_to_angles(sim_direction_to_real_direction(actions))
            # print(joint_angles)
            set_joint_angles(joint_angles)
            
            observation_encoder.hidden_states = (observation_encoder.hidden_states[0].detach(), observation_encoder.hidden_states[1].detach())
            last_inference_time = time.time()
            time_history.append(last_inference_time)
            i += 1
        else:
            observer.update_mpu_obs()
    print("------------------------\n**  ENDING ROLLOUT **\n------------------------\n")
    print(f"Actual time elapsed: {time.time() - pre_time}; target time elapsed: {ROLLOUT_S}")
    #observer.plot_history()
    #log = np.concatenate( (observer.history, np.array(time_history)[:, np.newaxis], np.array(actions_history)), axis=1 )
    #np.save(policy_path.split("/")[-1].split(".")[0] + ".npy", log)


if __name__ == "__main__":
    reset_joints()
    main()
    try:
        pass
        #main() 
    except Exception as e:
        print(e)
        reset_joints()
    reset_joints()  

