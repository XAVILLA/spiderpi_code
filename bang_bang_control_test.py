from utils import *

n = 100
reset_joints()
for i in range(n):
    if i == n - 1:
        a = [policy_outputs_to_angles(0)] * 18
    elif i % 3 == 0:
        a = [policy_outputs_to_angles(2.09)] * 18
    else:
        a = [policy_outputs_to_angles(-2.09)] * 18
    set_joint_angles(a)
    