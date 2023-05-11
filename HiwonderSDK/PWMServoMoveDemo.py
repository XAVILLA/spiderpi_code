import time
import Board

print('''
**********************************************************
********Function:Hiwonder SpiderPi expansion board, PWM servo control routine********
**********************************************************
----------------------------------------------------------
Official website:http://www.hiwonder.com
Online mall:https://huaner.tmall.com/
----------------------------------------------------------
The following commands need to be used in the LX terminal, which can be opened by ctrl+alt+t, or click
Click the black LX terminal icon in the upper bar
----------------------------------------------------------
Usage:
    sudo python3 BusServoMove.py
----------------------------------------------------------
Version: --V1.1  2020/11/07
----------------------------------------------------------
Tips:
 * Press Ctrl+C to close the program, if it fails, please try multiple times!
----------------------------------------------------------
''')

while True:
    # parameter：parameter 1：Servo port number; parameter 2: position; parameter 3: running time
    Board.setPWMServoPulse(2, 1500, 500)  # the servo NO.2 turn to 1500 position,it takes 500ms
    time.sleep(0.5)  # delay time is the same as running time
    
    Board.setPWMServoPulse(2, 1800, 500)  # The rotation range of the servo is 0-180 degrees, and the corresponding pulse width is 500-2500, that is, the range of parameter 2 is 500-2500
    time.sleep(0.5)
    
    Board.setPWMServoPulse(2, 1500, 200)
    time.sleep(0.2)
    
    Board.setPWMServoPulse(2, 1800, 500)  
    Board.setPWMServoPulse(1, 1800, 500)
    time.sleep(0.5)
    
    Board.setPWMServoPulse(2, 1500, 500)  
    Board.setPWMServoPulse(1, 1500, 500)
    time.sleep(0.5)    
