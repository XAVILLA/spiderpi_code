import time
import threading
import ActionGroupControl as AGC

print('''
**********************************************************
*********Function: Hiwonder SpiderPi expansion board, action group control routine********
**********************************************************
----------------------------------------------------------
Official website:http://www.hiwonder.com
Online mall:https://huaner.tmall.com/
----------------------------------------------------------
The following commands need to be used in the LX terminal, which can be opened by ctrl+alt+t, or click
Click the black LX terminal icon in the upper bar.
----------------------------------------------------------
Usage:
    sudo python3 ActionGroupControlDemo.py
----------------------------------------------------------
Version: --V1.2  2020/12/23
----------------------------------------------------------
Tips:
 * Press Ctrl+C to close the program, if it fails, please try multiple times！
----------------------------------------------------------
''')

# the action group needs to be saved in the/home/pi/SpiderPi/ActionGroups
AGC.runActionGroup('go_forward_low', times=50)  # Parameter is action group, without a suffix and passed in as a character
# AGC.runActionGroup('turn_right_low', times=50)  # The second parameter running actions times, the default is 1. When 0, it indicates cyclic running. The third parameter indicates whether to stop at attention at the end

# time.sleep(1)

# th = threading.Thread(target=AGC.runActionGroup, args=('turn_right_low', 0), daemon=True)  # Running action functions is blocking. If you want to loop for a period of time and then stop, start with a thread
# th.start()
# time.sleep(3)
# AGC.stopAction()  # After 3s and stop sending command
# while th.is_alive(): # Wait the action stop
#     time.sleep(0.01)
# th = threading.Thread(target=AGC.runActionGroup, args=('turn_right_low', 0), daemon=True)  # Running action functions is blocking. If you want to loop for a period of time and then stop, start with a thread
# th.start()
# time.sleep(2)
# AGC.stopAction()  # After 3s and stop sending command
# while th.is_alive(): # Wait the action stop
#     time.sleep(0.01)
# AGC.runActionGroup('go_forward_low', times=20)  # The second parameter running actions times, the default is 1. When 0, it indicates cyclic running. The third parameter indicates whether to stop at attention at the end

# time.sleep(1)    
# AGC.runActionGroup('stand_low')
