#!/usr/bin/env python3
import os
import sys
import time
import pigpio
import RPi.GPIO as GPIO
from PWMServo import *
from BusServoCmd import *
from smbus2 import SMBus, i2c_msg

# Hiwonder raspberrypi extension sdk#
if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

Servos = ()
pi = pigpio.pi()
def initPWMServo(d):
    global Servos
    
    servo1 = PWM_Servo(pi, 12, deviation=d[0], control_speed=True)  # expansion board 1
    servo2 = PWM_Servo(pi, 13, deviation=d[1], control_speed=True)  # 2
    Servos = (servo1, servo2)

d = [0, 0]
initPWMServo(d)

def setPWMServoPulse(servo_id, pulse = 1500, use_time = 1000):
    if servo_id < 1 or servo_id > 2:
        return

    pulse = 2500 if pulse > 2500 else pulse    
    pulse = 500 if pulse < 500 else pulse 
    use_time = 30000 if use_time > 30000 else use_time    
    use_time = 20 if use_time < 20 else use_time

    Servos[servo_id - 1].setPosition(pulse, use_time)

    return pulse

def setDeviation(servo_id, dev):
    if servo_id < 1 or servo_id > 2:
        return
    if d < -300 or d > 300:
        return
    Servos[servo_id - 1].setDeviation(dev)

def setBuzzer(new_state):
    GPIO.setup(31, GPIO.OUT)
    GPIO.output(31, new_state)

def setBusServoID(oldid, newid):
    """
    configure the servo ID, the factory default is 1
    :param oldid: the original id, the factory default is 1
    :param newid: the new id
    """
    serial_serro_wirte_cmd(oldid, LOBOT_SERVO_ID_WRITE, newid)

def getBusServoID(id=None):
    """
    read serial servo id
    :param id: default is empty
    :return: return servo id
    """
    
    while True:
        if id is None:  # there can only be one steering gear on the bus
            serial_servo_read_cmd(0xfe, LOBOT_SERVO_ID_READ)
        else:
            serial_servo_read_cmd(id, LOBOT_SERVO_ID_READ)
        # get content
        msg = serial_servo_get_rmsg(LOBOT_SERVO_ID_READ)
        if msg is not None:
            return msg

def setBusServoPulse(id, pulse, use_time):
    """
    driver serial servo rotation to designation position
    :param id: need to driver servo id
    :pulse: position
    :use_time: time required for rotation
    """

    pulse = 0 if pulse < 0 else pulse
    pulse = 1000 if pulse > 1000 else pulse
    use_time = 0 if use_time < 0 else use_time
    use_time = 30000 if use_time > 30000 else use_time
    serial_serro_wirte_cmd(id, LOBOT_SERVO_MOVE_TIME_WRITE, pulse, use_time)

def stopBusServo(id=None):
    '''
    stop servo run
    :param id:
    :return:
    '''
    serial_serro_wirte_cmd(id, LOBOT_SERVO_MOVE_STOP)

def setBusServoDeviation(id, d=0):
    """
    deviation
    :param id: servo id
    :param d:  deviation
    """
    serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_ADJUST, d)

def saveBusServoDeviation(id):
    """
    configuration deviation, power failure protection
    :param id: servo id
    """
    serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_WRITE)

time_out = 50
def getBusServoDeviation(id):
    '''
    read deviation value
    :param id: servo number
    :return:
    '''
    # send read deviation command
    count = 0
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_READ)
        # read
        msg = serial_servo_get_rmsg(LOBOT_SERVO_ANGLE_OFFSET_READ)
        count += 1
        if msg is not None:
            return msg
        if count > time_out:
            return None

def setBusServoAngleLimit(id, low, high):
    '''
    set servo rotation range
    :param id:
    :param low:
    :param high:
    :return:
    '''
    serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_LIMIT_WRITE, low, high)

def getBusServoAngleLimit(id):
    '''
    read servo rotation range
    :param id:
    :return: return 0： low position  1： high position
    '''
    
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_ANGLE_LIMIT_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_ANGLE_LIMIT_READ)
        if msg is not None:
            count = 0
            return msg

def setBusServoVinLimit(id, low, high):
    '''
    servo servo voltage range
    :param id:
    :param low:
    :param high:
    :return:
    '''
    serial_serro_wirte_cmd(id, LOBOT_SERVO_VIN_LIMIT_WRITE, low, high)

def getBusServoVinLimit(id):
    '''
    read servo rotation range
    :param id:
    :return: return 0： low position  1： high position
    '''
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_VIN_LIMIT_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_VIN_LIMIT_READ)
        if msg is not None:
            return msg

def setBusServoMaxTemp(id, m_temp):
    '''
    set the highest temperature alarm of the servo
    :param id:
    :param m_temp:
    :return:
    '''
    serial_serro_wirte_cmd(id, LOBOT_SERVO_TEMP_MAX_LIMIT_WRITE, m_temp)

def getBusServoTempLimit(id):
    '''
    read servo temperature alarm range
    :param id:
    :return:
    '''
    
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_TEMP_MAX_LIMIT_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_TEMP_MAX_LIMIT_READ)
        if msg is not None:
            return msg

def getBusServoPulse(id):
    '''
    read servo current position
    :param id:
    :return:
    '''
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_POS_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_POS_READ)
        if msg is not None:
            return msg

def getBusServoTemp(id):
    '''
    read servo temperature
    :param id:
    :return:
    '''
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_TEMP_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_TEMP_READ)
        if msg is not None:
            return msg

def getBusServoVin(id):
    '''
    read servo voltage
    :param id:
    :return:
    '''
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_VIN_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_VIN_READ)
        if msg is not None:
            return msg

def restBusServoPulse(oldid):
    # servo clear deviation and P value median(500)
    serial_servo_set_deviation(oldid, 0)    # erase deviation
    time.sleep(0.1)
    serial_serro_wirte_cmd(oldid, LOBOT_SERVO_MOVE_TIME_WRITE, 500, 100)    # middle position

## power down
def unloadBusServo(id):
    serial_serro_wirte_cmd(id, LOBOT_SERVO_LOAD_OR_UNLOAD_WRITE, 0)

## read if power down
def getBusServoLoadStatus(id):
    while True:
        serial_servo_read_cmd(id, LOBOT_SERVO_LOAD_OR_UNLOAD_READ)
        msg = serial_servo_get_rmsg(LOBOT_SERVO_LOAD_OR_UNLOAD_READ)
        if msg is not None:
            return msg
