import time
import Board

print('''
**********************************************************
********Function:Hiwonder SpiderPi expansion board, serial servo control routine*******
**********************************************************
----------------------------------------------------------
Official website:http://www.hiwonder.com
Online mall:https://huaner.tmall.com/
----------------------------------------------------------
The following commands need to be used in the LX terminal, which can be opened by ctrl+alt+t, or click
Click the black LX terminal icon in the upper bar.
----------------------------------------------------------
Usage:
    sudo python3 BusServoMove.py
----------------------------------------------------------
Version: --V1.1  2020/11/13
----------------------------------------------------------
Tips:
 * Press Ctrl+C to close the program, if it fails, please try multiple times！
----------------------------------------------------------
''')

while True:
    # Parameter：Parameter1：servo id：parameter1：servo id; parameter 2：position; parameter3：running time
    Board.setBusServoPulse(9, 200, 1000)  
    Board.setBusServoPulse(7, 500, 1000)
    time.sleep(1) 
    
    Board.setBusServoPulse(9, 500, 500)  # the servo NO.8 turn to 500 position,it takes 500ms
    time.sleep(0.5)  # delay time is the same as running time
    
    Board.setBusServoPulse(9, 800, 500)  # the rotation range of the servo is 0-240°，and thecorresponding pulse width is 0-1000,that is, the range of parameter 2 is 0-1000
    time.sleep(0.5)
    
    Board.setBusServoPulse(7, 300, 400)
    time.sleep(0.4)

    Board.setBusServoPulse(7, 700, 800)
    time.sleep(0.8) 

    Board.setBusServoPulse(7, 500, 400)
    time.sleep(0.4)  

   
