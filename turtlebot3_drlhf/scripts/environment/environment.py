#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import copy
import math
import rospy
import time
import numpy as np

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
import tf
import tf.transformations
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from squaternion import Quaternion
from std_msgs.msg import ColorRGBA
from gazebo_msgs.srv import SpawnModel, DeleteModel

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from common.settings import UNKNOWN, SUCCESS, COLLISION, \
    THRESHOLD_COLLISION, THREHSOLD_GOAL, ARENA_LENGTH, ARENA_WIDTH, \
    ORIGIN_POSE_X, ORIGIN_POSE_Y, MAX_LIN_VEL, MAX_ANG_VEL


TARGET_COLOR = ColorRGBA(1.0, 0.0, 0.0, 1.0)

class DRLEmvironment:
    def __init__(self):
        # rospy.init_node('turtlebot3_drl_env', anonymous=True)
        self.rate = rospy.Rate(100)
        
        self.scan_topic = '/scan'
        self.vel_topic = '/cmd_vel'
        self.odom_topic = '/odom'

        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_yaw = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0

        self.laser_data = []
        self.vel = Twist()

        self.entity_path = '/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/models/turtlebot3_drl_world/goal_box/model.sdf'
        self.entity = open(self.entity_path, 'r').read()

        # parameters of the environment
        self.state_dim = 52
        self.action_dim = 2
        self.observation_space = np.zeros(self.state_dim)
        self.action_space = np.zeros(self.action_dim)

        # flags
        self.done = True
        self.succeed = UNKNOWN

        # rewards
        self.reward = 0.0
        self.distance_reward = 0.0
        self.laser_reward = 0.0
        self.collision_reward = 0.0
        self.linear_punish_reward = 0.0
        self.angular_punish_reward = 0.0

        self.human_control = False
        self.goal_model_exist = False

        # tf listener
        self.tf_listener = tf.TransformListener()
        try:
            self.tf_listener.waitForTransform('odom', 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform('odom', 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.logerr('Cannot find transform between base_link and odom')
                rospy.signal_shutdown('tf Exception')

        # publishers
        self.vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=1)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.goal_point_maker_pub = rospy.Publisher('goal_point', Marker, queue_size=3)

        # subscribers
        self.laser_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.laser_callback)
        self.vel_sub = rospy.Subscriber(self.vel_topic, Twist, self.vel_callback)
        self.goal_pose_sub = rospy.Subscriber('/goal_pose', ModelState, self.goal_pose_callback)

        #serivces
        self.rest_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        #client spawn_entity
        self.spawn_entity = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        self.delet_entity = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        

    def get_odom(self):
        # update the robot's position and orientation
        try:
            (tran, rot) = self.tf_listener.lookupTransform('odom', self.base_frame, rospy.Time(0))
            rotation = tf.transformations.euler_from_quaternion(rot)
        
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.logerr('TF Exception')
            return None

        return (tran[0], tran[1], rotation[2])
    
    def goal_pose_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        quat = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quat)
        self.goal_yaw = euler[2]
        rospy.loginfo("Goal pose received: x: %f, y: %f, yaw: %f" % (self.goal_x, self.goal_y, self.goal_yaw))

    def vel_callback(self, msg):   
        self.vel = msg

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

    def collision_check(self, laser_values):
        collision_reward = 0
        for i in range(len(laser_values)):
            if (laser_values[i] < 2*THRESHOLD_COLLISION):
                collision_reward = -80
            if (laser_values[i] < THRESHOLD_COLLISION):
                collision_reward = -2000
                self.succeed = COLLISION
                break
        return collision_reward
    
    # generate a red circle in gazebo
    def generate_goal_model(self, goal_x, goal_y):
        goal_pose = Pose()
        goal_pose.position.x = goal_x
        goal_pose.position.y = goal_y

        #call the service to spawn the entity
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        try:
            resp = self.spawn_entity(model_name='goal_box', model_xml=self.entity, robot_namespace='', initial_pose=goal_pose, reference_frame='world')
            if resp.success:
                self.goal_model_exist = True
        
        except rospy.ServiceException as e:
            rospy.logerr('Spawn Model service call failed')

    def delete_goal_model(self):
        rospy.wait_for_service('gazebo/delete_model')
        try:
            resp = self.delet_entity(model_name='goal_box')
            if resp.success:
                self.goal_model_exist = False
                
        except rospy.ServiceException as e:
            rospy.logerr('Delete Model service call failed')

    def reset(self):
        self.succeed = UNKNOWN
        self.vel_pub.publish(Twist())
        self.done = True
        if self.goal_model_exist:
            self.delete_goal_model()

        # reset the robot's position
        # try:
        #     self.rest_proxy()
        # except rospy.ServiceException as e:
        #     rospy.logerr('Reset Simulation failed')

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        state_msg = ModelState()    
        state_msg.model_name = 'robot'
        state_msg.pose.position.x = ORIGIN_POSE_X
        state_msg.pose.position.y = ORIGIN_POSE_Y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quaternion.x
        state_msg.pose.orientation.y = quaternion.y
        state_msg.pose.orientation.z = quaternion.z
        state_msg.pose.orientation.w = quaternion.w

        try:
            self.set_state.publish(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr('Set Model State failed')

        self.robot_x = state_msg.pose.position.x
        self.robot_y = state_msg.pose.position.y

        self.goal_x = np.random.uniform(-ARENA_LENGTH, ARENA_LENGTH)
        self.goal_y = np.random.uniform(-ARENA_WIDTH, ARENA_WIDTH)
        self.goal_yaw = np.random.uniform(-math.pi, math.pi)

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        
        self.goal_point_maker_pub.publish(marker)

        if not self.goal_model_exist:
            self.generate_goal_model(self.goal_x, self.goal_y)

        angle_tartget    = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        angle_diff = angle_tartget - self.robot_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        distance = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        normalized_laser = [(x)/3.5 for x in (self.laser_data)]

        state = copy.deepcopy(normalized_laser)
        state.append(float(distance))
        state.append(float(angle_diff))
        state.append(0.0)
        state.append(0.0)
        state = np.array(state)

        return state
    
    def read_action(self):
        action = []
        action.append(self.vel.linear.x)
        action.append(self.vel.angular.z)
        action = np.array(action)

        return action

    
    def step(self, action = None, time_duration = 0.05):
        
        (self.robot_x, self.robot_y, self.robot_yaw) = self.get_odom()
        robot_x_prev, robot_y_prev = self.robot_x, self.robot_y

        vel_cmd = Twist()
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < time_duration:
            if not self.human_control:
                if action is not None:
                    vel_cmd.linear.x = action[0]*MAX_LIN_VEL
                    vel_cmd.angular.z = action[1]*MAX_ANG_VEL
                self.vel_pub.publish(vel_cmd)
            self.rate.sleep()
        
        (self.robot_x, self.robot_y, self.robot_yaw) = self.get_odom()

        angle_target = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        angle_diff = angle_target - self.robot_yaw
        while  angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        normalized_laser = [np.clip(float(x)/3.5, 0, 1) for x in self.laser_data]
        distance = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

        state = copy.deepcopy(normalized_laser)
        state.append(float(distance))
        state.append(float(angle_diff))
        state.append(action[0])
        state.append(action[1])
        state = np.array(state)

        distance_prev = math.hypot(self.goal_x - robot_x_prev, self.goal_y - robot_y_prev)
        self.distance_reward = distance_prev - distance
        self.laser_reward = sum(normalized_laser)-24
        self.collision_reward = self.collision_check(self.laser_data) + self.laser_reward 
        self.vel_reward = -1 * (((0.15 - action[0]) * 10) ** 2) + -1 * (action[1]**2)
        self.reward = self.distance_reward + self.collision_reward + self.vel_reward

        if self.done:
            self.done = False

        if distance < THREHSOLD_GOAL:
            self.succeed =  SUCCESS
            self.reward += 2500

        if self.succeed != UNKNOWN:
            self.done = True
            self.reset()
            rospy.sleep(1)
            

        return state, self.reward, self.done, self.succeed
    
    
if __name__ == '__main__':
    
    env = DRLEmvironment()
    while not rospy.is_shutdown():
        state = env.reset()
        time.sleep(5)

    








