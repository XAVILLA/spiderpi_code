import time
import numpy as np
import math
import torch
from observer import Observer
from observer_save_depth import Observer_onlydepth
from high_level_observer import HighLevelObserver
from matplotlib import pyplot as plt
from utils import *

from actor_critic_recurrent import ActorCriticRecurrent


def hl_features(hl_observation):
    img_copy = hl_observation.copy()
    img_h, img_w = hl_observation.shape[:2]

    blob = cv2.dnn.blobFromImage(img_copy, 1, (250, 250), [104, 117, 123], False, False)
    
    #net.setInput(blob)
    #detections = net.forward() # Calculated recognition
    
    return img


def main():
    perception = False
    use_prior = False
    policy_path = "/home/pi/Desktop/hexapod-isaac/sim_to_real/Jason's models/"
    policy_path += "rough_spiderpi_Nov22_00-15-31_finetune_patterned_random_correct_terrain_phase_two_yaw.pt"
    #policy_path += "rough_spiderpi_Dec13_16-32-13_blind_withmount_patternedjoist_phase_two_yaw.pt"
    checkpoint = torch.load(policy_path, map_location=torch.device("cpu"))
    num_obs = 3+18
    if perception:
        num_obs += 16*16
    if use_prior:
        num_obs += 4 + 2
    if use_prior:
        num_actions = 22
        max_action = 1.0
    else:
        num_actions = 18
        max_action = 2.094
#     action_scale = 1/64 #0.25
    action_scale = 1/5
    actor_critic = ActorCriticRecurrent(
        num_obs,
        num_obs,
        num_actions,
        actor_hidden_dims=[128, 64, 32], #[512, 256, 128],
        rnn_type='lstm',
        rnn_hidden_size=64, #512,
        rnn_num_layers=1
    )
    if perception:
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
    observer = Observer()

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
            observation = [observation]
            if use_prior:
                observation.append( torch.sin(torch.Tensor([phi])) )
                observation.append( torch.cos(torch.Tensor([phi])) )
            observation.append(prev_actions)
            observation = torch.cat(observation, dim=-1).unsqueeze(0)
            z_t = observation_encoder(observation)
            actions = base_policy(z_t)
            actions = torch.clip(actions, min=-max_action / action_scale, max=max_action / action_scale)[0, 0]
            prev_actions = actions.clone() # save unscaled actions as prev actions as in sim
            actions = actions.detach().numpy()
            if use_prior:
                actions *= action_scale
                
                #actions *= 0
                
                residuals, f_tg, a1, a2, a3 = actions[:18], actions[18], actions[19], actions[20], actions[21]
                
                #f_tg += 0.5
                #a1 += 0.3
                # a2 += 0.35
                #a3 += 0.25
                
                residuals *= 0.1
                f_tg = f_tg*0.625 + 0.625
                a1 *= 2.094
                a2 *= 2.094
                a3 *= 2.094
                phi_leg = (phi + delta_phi_leg) % (2*np.pi)
                c = np.cos(phi_leg)
                j1 = a1*c
                j2 = a2*c
                j3 = a3*c
                targets = []
                for j in range(6):
                    targets.append(j1[j])
                    targets.append(j2[j])
                    targets.append(j3[j])
                actions = np.array(targets) + residuals
                temp_actions = np.copy(actions)
                actions[9:12] = temp_actions[15:18]# reorder some legs from sim order to real order
                actions[15:18] = temp_actions[9:12]
                phi = phi + 2*np.pi*f_tg*DELTA_T
            else:
                # stuff that only has to happen on the robot
                actions[9:12] = prev_actions[15:18].detach().numpy() # reorder some legs from sim order to real order
                actions[15:18] = prev_actions[9:12].detach().numpy()
                actions *= action_scale # scale down
            # actions *= np.array([1, 0.73, 0.73] * 6) # scale down joint 1 on all legs to avoid self collision
            actions_history.append(actions)
            
            joint_angles = policy_outputs_to_angles(sim_direction_to_real_direction(actions))
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

