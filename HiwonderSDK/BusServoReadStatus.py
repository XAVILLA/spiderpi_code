import time
import Board

print('''
**********************************************************
*****Function:Hiwonder SpiderPi expansion board, serial servo control routine******
**********************************************************
----------------------------------------------------------
Official website:http://www.hiwonder.com
Online mall:https://huaner.tmall.com/
----------------------------------------------------------
The following commands need to be used in the LX terminal, which can be opened by ctrl+alt+t, or click
Click the black LX terminal icon in the upper bar.
----------------------------------------------------------
Usage:
    sudo python3 BusServoReadStatus.py
----------------------------------------------------------
Version: --V1.1  2020/11/13
----------------------------------------------------------
Tips:
 * Press Ctrl+C to close the program, if it fails, please try multiple times！
----------------------------------------------------------
''')

def getBusServoStatus():
    Pulse = Board.getBusServoPulse(9)  # get the NO.9 servo postion information
    Temp = Board.getBusServoTemp(9)  # get the NO.9 servo temperature information
    Vin = Board.getBusServoVin(9)  # get the NO.9 servo voltage information
    print('Pulse: {}\nTemp:  {}℃\nVin:   {}mV\n'.format(Pulse, Temp, Vin)) # print statue information
    time.sleep(0.5)  # delay easy to check

while True:   
    Board.setBusServoPulse(9, 500, 1000) # the servo NO.9 turn to 500 position,it takes 1000ms
    time.sleep(1)
    getBusServoStatus()
    Board.setBusServoPulse(9, 300, 1000)
    time.sleep(1)
    getBusServoStatus()
