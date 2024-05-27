#!/usr/bin/env python3

from pynput import keyboard
import threading
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import rospy
import sys
import tty
import termios
from select import select

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
        w    
   a    s    d

CTRL-C to quit
"""

class KeyboardTeleop:
    def __init__(self):
        self.pressed_list = set()
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.done = False
        self.keyboard_listener.start()

    def on_press(self, key):
        if (key == '\x03'):
            self.done = True
        else:
            self.pressed_list.add(key)

    def on_release(self, key):
        if key != '\x03':
            try:
                self.pressed_list.remove(key.char)
            except AttributeError:
                self.pressed_list.remove(key)

    def get_key(self):
        return self.pressed_list
    
def main():
    rospy.init_node('teleop_by_pynput')
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    teleop = KeyboardTeleop()
    twist = Twist()
    print(msg)
    while not rospy.is_shutdown():
        key = teleop.get_key()
        x = int('w' in key) - int('s' in key)
        z = int('a' in key) - int('d' in key)
        twist.linear.x = WAFFLE_MAX_LIN_VEL * x 
        twist.angular.z = WAFFLE_MAX_ANG_VEL * z

        publisher.publish(twist)
        if teleop.done:
            break

    publisher.publish(Twist())

if __name__ == '__main__':
    main()

        
    






