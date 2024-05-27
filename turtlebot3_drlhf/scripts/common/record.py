#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from environment.env import DRLHFEnv
import scipy.io as sio
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import os
import sys
import subprocess

class Recorder:
    def __init__(self):
        self.records = []
        self.rewards = []
        self.env = DRLHFEnv()
        self.env.reset()
        self.env.human_control = True

    def remeber(self, state, action, reward, next_state, done):
        self.records.append((state, action, reward, next_state, done))
    
    def save_records(self):
        print(self.__len__())
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for s,a,r,n,d in self.records:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)
            dones.append(d)
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}

        # records plus rospy.Time.now().to_sec() to avoid overwriting
        filename = '/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/records/records_{}.mat'.format(rospy.Time.now().to_sec())
        sio.savemat(filename, data)

    def reset(self):
        self.records = []

    def __len__(self):
        return len(self.records)

class HumanControl:
    def __init__(self):
        
        self.joy_msg = None
        self.factor_linear = 0.26
        self.factor_angular = 1.82
        self.linear_speed = 0
        self.angular_speed = 0
        self.start_flag = 0
        self.restart_flag = 0
        self.cancel_flag = 0
        self.exit_flag = 0
        self.save_flag = 0
        self.recorder = Recorder()
        self.recorder.reset()
        self.process = None

        device_file = '/dev/input/js0'
        if not os.path.exists(device_file):
            rospy.logerr("Joystick device not found")
            sys.exit()
        else:
            #use python rosrun joy joy_node
            rospy.loginfo("Joystick device found")
            self.process = subprocess.Popen(['rosrun', 'joy', 'joy_node'])


        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


    def joy_callback(self, joy_msg):
        self.linear_speed = joy_msg.axes[1]*self.factor_linear
        self.angular_speed = joy_msg.axes[2]*self.factor_angular
        self.start_flag = joy_msg.buttons[6] if self.start_flag == 0 else self.start_flag
        self.cancel_flag = joy_msg.buttons[7] if self.cancel_flag == 0 else self.cancel_flag
        self.save_flag = joy_msg.buttons[0]
        if self.save_flag == 1:
            self.recorder.save_records()
            self.recorder.reset()
            rospy.loginfo("file saved")
            rospy.loginfo("Press Start Button to Start Game or Press Cancel Button to exit")
            self.save_flag = 0
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.vel_pub.publish(twist)
        
    def get_key(self):
        return self.button_select

    def reset(self):
        self.button_select = 0

    def run(self):
        while not rospy.is_shutdown():
            if len(self.recorder.records) > 0:
                rospy.loginfo("Press save button to save records or press start button to start new game")
            else:
                rospy.loginfo("Press Start Button to Start Game")
            
            while self.start_flag == 0 and self.cancel_flag == 0:
                rospy.sleep(0.1)
            if self.cancel_flag == 1:
                rospy.loginfo("Game Canceled")
                self.cancel_flag = 0
                break
            if self.start_flag == 1:
                rospy.loginfo("Game Started")
                self.start_flag = 0

            state = self.recorder.env.reset()
            while True:
                action = self.recorder.env.read_action()
                next_state, reward, done, succeed = self.recorder.env.step(action)
                self.recorder.remeber(state, action, reward, next_state, done)
                state = next_state
                if done:
                    rospy.loginfo("Game Over")
                    break

        self.process.terminate()
        rospy.signal_shutdown("Exit Game")
        
                        
if __name__ == '__main__':
    rospy.init_node('human_control')
    human_control = HumanControl()
    human_control.run()
    rospy.spin()
    # main()

    
                
                
                

    

    


    
            

    



