from HiwonderSDK import Board
import time

def get_dof_pos():
    pre = time.time()
    dof_pos = []
    for i in range(1, 19):
        joint_pre = time.time()
        if i != 9:
            dof_pos.append(Board.getBusServoPulse(i))
        print(f"{i}: {time.time() - joint_pre}")
    return dof_pos, time.time() - pre

if __name__ == "__main__":
    num_trials = 100
    times = []
    for _ in range(num_trials):
        _, t = get_dof_pos()
        times.append(t)
        print(t)
    print(f"Average read time for all joints: {sum(times) / len(times)}s.")
        