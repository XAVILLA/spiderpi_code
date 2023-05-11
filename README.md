After training a phase 2 perceptive model in simulation, the model checkpoint file gets saved to the sim_to_real directory in rl-legged-locomotion-isaac

### To run the perceptive model on spiderpi: ###

1. Transfer the model file from the training machine to raspi on spiderpi.
2. Change the policy path variable in sim_to_real_free_perceptive.py to be the path to your new model
3. Run python3 sim_to_real_free_perceptive.py to have the robot running!
4. To reset the joint position after a trail, run reset_joints.py

### Important scripts: ###

1. sim_to_real_free_perceptive.py: the script you use to run a perceptive model
2. sim_to_real_free.py: the script was written by Maxime to run a blind model
3. observer_perceptive.py: the script that collects perception and body orientation input to the model from the two mounted cameras. 
4. actor_critic_recurrent_cnn_s2r.py: the policy model file that’s exactly the same as the one used during training, in the modules folder of the rsl_rl repository. If you change the model, remember to put the model file on spiderpi and import that into sim_to_real_free_perceptive.py
5. utils.py: the file that has the API to set robot joint positions. 

### Things to note: ###
1. The tracking camera T265 needs to be power-cycled (unplug and replug) every time the raspi restarts (only when raspi restarts, not when every time you rerun the script), otherwise it can’t be detected. 
2. You can’t use cal_visitor wifi to ssh since it doesn’t support port 22. Use Berkeley_IoT instead. To ssh into the robot, do ssh pi@{ip_address}. To find the ip address, you can use a monitor to connect to the robot and check it. The current ip address is 10.142.18.87 (it might change). The password of the robot is spiderpi.
3. The best perceptive model I have is rough_perceptive_maxrange1m_legmask_invalidmaskcorrect_depthinvalidmask2.3.pt