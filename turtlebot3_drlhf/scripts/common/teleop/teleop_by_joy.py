#!/usr/bin/env python3
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import rospy
import numpy as np

class JoyTeleop:
    def __init__(self):
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.joy_msg = None
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.factor_linear = 0.5
        self.factor_angular = 1.82
        self.button_select = 0



    def joy_callback(self, joy_msg):
        # linear_speed = np.sqrt(joy_msg.axes[1]**2 + joy_msg.axes[0]**2)*self.factor_linear*np.sign(joy_msg.axes[1])
        # angular_speed = np.sqrt(joy_msg.axes[2]**2 + joy_msg.axes[3]**2)*self.factor_angular*np.sign(joy_msg.axes[2])
        linear_speed = joy_msg.axes[1]*self.factor_linear
        angular_speed = joy_msg.axes[2]*self.factor_angular
        self.button_select = joy_msg.buttons[10]

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.vel_pub.publish(twist)
        rospy.loginfo("linear:%f, angular:%f\n",twist.linear.x, twist.angular.z)
        
def main():
    rospy.init_node('joy_teleop')
    joy_teleop = JoyTeleop()
    rospy.spin()


if __name__ == '__main__':
    main()
