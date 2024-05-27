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
from gazebo_msgs.srv import SetModelState, GetModelState

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from common.settings import *

class DRLHFEnv:
    def __init__(self):
        
        self.rate = rospy.Rate(100)

        self.scan_topic = '/scan'
        self.cmd_vel_topic = '/cmd_vel'
        self.odom_topic = '/odom'
        
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_prev_x, self.robot_prev_y = 0.0, 0.0
        self.robot_yaw = 0.0

        self.goal_x, self.goal_y = 0.0, 0.0
        self.goal_yaw = 0.0
        self.goal_point_radius = THREHSOLD_GOAL
        self.goal_inintial_dist = 0.0

        self.pose_list = [[-0.744, 0.153],[0.36, 1.46],[1.68, -0.87],[0.0, -1.0],[-1.2, -1.0],[-0.75, 0.0],[-0.77, 0.86]]

        self.laser_data = None
        self.scan_ranges = [LIDAR_DISTANCE_CAP] * NUM_SCAN_SAMPLES
        self.obstacle_distance = LIDAR_DISTANCE_CAP
        self.vel = Twist()
        self.action_lin = 0.0
        self.action_ang = 0.0

        self.entity_path = '/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/models/turtlebot3_drl_world/goal_box/model.sdf'
        self.entity = open(self.entity_path, 'r').read()
        self.entity_exists = False

        # parameters of the DRL environment
        self.state_dim = NUM_SCAN_SAMPLES + 4
        self.action_dim = 2
        self.observation_space = np.zeros(self.state_dim)
        self.action_space = np.zeros(self.action_dim)

        #flag for the simulation
        self.done = False
        self.success = UNKNOWN

        # rewards
        self.reward = 0.0
        self.distance_reward = 0.0
        self.laser_reward = 0.0
        self.collision_reward = 0.0
        self.vel_reward = 0.0
        self.linear_punish_reward = 0.0
        self.angular_punish_reward = 0.0

        # Other parameters
        self.human_control = False

        """ ROS initialization """ 
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

        # ROS publishers
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=5)
        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.maker_pub = rospy.Publisher('goal_point_maker', Marker, queue_size=1)
        
        # ROS subscribers
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback)
        self.vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.vel_callback)

        # ROS services
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.spawn_entity = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        self.delete_entity = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        self.gazebo_pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.gazebo_unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_state = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        #spawn the goal point
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        try:
            resp = self.spawn_entity(model_name='goal_circle', model_xml=self.entity, robot_namespace='', initial_pose=Pose(), reference_frame='world')
            if resp.success:
                self.entity_exists = True
        
        except rospy.ServiceException as e:
            rospy.logerr('Spawn Model service call failed')

        # wait for the env to be ready
        rospy.loginfo('Waiting for the environment to be ready...')
        while self.scan_ranges is None:
            self.rate.sleep()

    ''' Callback functions '''

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

    def scan_callback(self, msg):
        self.laser_data = msg.ranges
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = np.clip(float(msg.ranges[i]) / LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= LIDAR_DISTANCE_CAP

    def vel_callback(self, msg):   
        self.vel = msg

    def read_action(self):
        if self.human_control:
            lin = float(self.vel.linear.x/MAX_LIN_VEL)
            ang = float(self.vel.angular.z/MAX_ANG_VEL)
            return [lin, ang]
        else:
            return None
        
    def model_exists(self, model_name):
        try:
            resp = self.get_model_state(model_name, '')
            if resp.success:
                return True
            else:
                return False
        except rospy.ServiceException as e:
            rospy.logerror("Service call failed: %s" % e)
            return False

    def generate_goal_point(self, goal_x, goal_y):
        ''' Generate a random goal point'''
        # Rviz marker
        goal_maker = Marker()
        goal_maker.header.frame_id = "odom"
        goal_maker.type = goal_maker.CYLINDER
        goal_maker.action = goal_maker.ADD
        goal_maker.scale.x = 0.1
        goal_maker.scale.y = 0.1
        goal_maker.scale.z = 0.01
        goal_maker.color.a = 1.0
        goal_maker.color.r = 0.0
        goal_maker.color.g = 1.0
        goal_maker.color.b = 0.0
        goal_maker.pose.orientation.w = 1.0
        goal_maker.pose.position.x = goal_x
        goal_maker.pose.position.y = goal_y
        goal_maker.pose.position.z = 0

        self.maker_pub.publish(goal_maker)

        # Spawn the goal point model
        goal_pose = Pose()
        goal_pose.position.x = goal_x
        goal_pose.position.y = goal_y
        goal_pose.position.z = 0.0

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_resp = self.spawn_entity(model_name='goal_circle', model_xml=self.entity, robot_namespace='', initial_pose=goal_pose, reference_frame='world')
            rospy.loginfo(spawn_resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn SDF Model service call failed: %s" % e)

    def remove_goal_point(self):
        ''' Remove the goal point entity '''
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_resp = self.delete_entity('goal_circle')
            rospy.loginfo(delete_resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("Delete Model service call failed: %s" % e)

    def pause_sim(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.gazebo_pause()
        except rospy.ServiceException as e:
            rospy.logerr('Pause Physics service call failed')
    
    def unpause_sim(self):
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.gazebo_unpause()
        except rospy.ServiceException as e:
            rospy.logerr('Unpause Physics service call failed')

    def reward_A(self, success, action_linear, action_angular, distance, goal_angle, min_obstacle_distance):
        reward = 0
        r_yaw = -10.0 * abs(goal_angle)
        r_angular = -1 * (action_angular**2)
        r_distance = (2 * self.goal_inintial_dist) / (self.goal_inintial_dist + distance) - 1
        r_linear = -1 * (((0.26 - action_linear) * 10) ** 2)

        if min_obstacle_distance < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0
        
        reward = r_yaw + r_distance + r_obstacle + r_angular + r_linear - 1
        if success == SUCCESS:
            reward += 2500
        elif success == COLLISION:
            reward -= 2000

        return float(reward)

    def reward_B(self, success, distance, action_linear, action_angular, laser_data, min_obstacle_distance):

        distance_reward = distance * (5 / 0.02) * 1.2 * 7
        laser_reward = sum(laser_data) - NUM_SCAN_SAMPLES
        if min_obstacle_distance < THRESHOLD_COLLISION:
            collision_reward = -200.0
        elif min_obstacle_distance < 2 * THRESHOLD_COLLISION:
            collision_reward = -80.0

        if action_angular > 0.8 or action_angular < -0.8:
            vel_reward = -1
        if action_linear < 0.2:
            vel_reward -= 2

        reward = distance_reward + laser_reward + collision_reward + vel_reward

        if success == SUCCESS:
            reward += 100
        elif success == COLLISION:
            reward -= 200
        
        return float(reward)
    
    def set_env(self):

        self.done = True
        self.success = UNKNOWN
        self.vel_pub.publish(Twist())
        if self.model_exists('goal_circle'):
            self.remove_goal_point()

        angle = np.pi
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        state_msg = ModelState()    
        state_msg.model_name = 'robot'
        state_msg.pose.position.x = -1.0
        state_msg.pose.position.y = -1.0
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quaternion.x
        state_msg.pose.orientation.y = quaternion.y
        state_msg.pose.orientation.z = quaternion.z
        state_msg.pose.orientation.w = quaternion.w

        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr('Set Model State failed')
        
        self.robot_x = -1.0
        self.robot_y = -1.0
        
        self.goal_x = 1.0
        self.goal_y = 1.0
        self.goal_yaw = np.random.uniform(-math.pi, math.pi)

        time = 0
        while not self.model_exists('goal_circle'):
            self.generate_goal_point(self.goal_x, self.goal_y)
            rospy.sleep(1.0)
            time += 1
            if time > 5:
                rospy.logerr('Failed to spawn the goal point, pls restart the simulation')
                rospy.signal_shutdown('Spawn exception')
                break
                
        angle_tartget = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        angle_diff = angle_tartget - self.robot_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        distance = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        self.goal_inintial_dist = distance
        normalized_laser = self.scan_ranges
        
        state = copy.deepcopy(normalized_laser)
        state.append(float(distance))
        state.append(float(angle_diff))
        state.append(0.0)
        state.append(0.0)

        return state

        
    def reset(self):
        ''' Reset the environment '''
        self.done = True
        self.success = UNKNOWN
        self.vel_pub.publish(Twist())
        if self.model_exists('goal_circle'):
            self.remove_goal_point()

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        index = np.random.randint(0, len(self.pose_list))
        state_msg = ModelState()    
        state_msg.model_name = 'robot'
        state_msg.pose.position.x = self.pose_list[index][0]
        state_msg.pose.position.y = self.pose_list[index][1]
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = quaternion.x
        state_msg.pose.orientation.y = quaternion.y
        state_msg.pose.orientation.z = quaternion.z
        state_msg.pose.orientation.w = quaternion.w

        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr('Set Model State failed')
        
        self.robot_x = self.pose_list[index][0]
        self.robot_y = self.pose_list[index][1]
        
        self.goal_x = np.random.uniform(-ARENA_LENGTH, ARENA_LENGTH)
        self.goal_y = np.random.uniform(-ARENA_WIDTH, ARENA_WIDTH)
        self.goal_yaw = np.random.uniform(-math.pi, math.pi)

        time = 0
        while not self.model_exists('goal_circle'):
            self.generate_goal_point(self.goal_x, self.goal_y)
            rospy.sleep(1.0)
            time += 1
            if time > 5:
                rospy.logerr('Failed to spawn the goal point, pls restart the simulation')
                rospy.signal_shutdown('Spawn exception')
                break
                
        angle_tartget = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        angle_diff = angle_tartget - self.robot_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        distance = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        self.goal_inintial_dist = distance
        normalized_laser = self.scan_ranges
        
        state = copy.deepcopy(normalized_laser)
        state.append(float(distance))
        state.append(float(angle_diff))
        state.append(0.0)
        state.append(0.0)

        return state
    
    def step(self, action = None, time_duration = 0.02):

        (self.robot_x, self.robot_y, self.robot_yaw) = self.get_odom()
        robot_x_prev, robot_y_prev = self.robot_x, self.robot_y

        vel_cmd = Twist()
        start_time = rospy.Time.now().to_sec()
        
        while rospy.Time.now().to_sec() - start_time < time_duration:
            if not self.human_control:
                if action is not None:
                    if ENABLE_BACKWARD:
                        self.action_lin = action[0] * MAX_LIN_VEL
                    else:
                        self.action_lin = (action[0] + 1.0) / 2 * MAX_LIN_VEL
                    self.action_ang = action[1] * MAX_ANG_VEL
                    vel_cmd.linear.x = self.action_lin
                    vel_cmd.angular.z = self.action_ang
                else:
                    rospy.logerr('Action is None')
                    
                self.vel_pub.publish(vel_cmd)
                self.rate.sleep()
            else:
                vel_cmd.linear.x = self.vel.linear.x
                vel_cmd.angular.z = self.vel.angular.z

        (self.robot_x, self.robot_y, self.robot_yaw) = self.get_odom()
        
        # self.pause_sim()

        angle_target = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        angle_diff = angle_target - self.robot_yaw
        while  angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        normalized_laser = self.scan_ranges
        distance = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

        state = copy.deepcopy(normalized_laser)
        state.append(float(distance))
        state.append(float(angle_diff))
        state.append(vel_cmd.linear.x)
        state.append(vel_cmd.angular.z)

        distance_prev = math.hypot(self.goal_x - robot_x_prev, self.goal_y - robot_y_prev)

        if self.done:
            self.done = False

        if self.obstacle_distance < THRESHOLD_COLLISION:
            self.success = COLLISION
        
        if distance < THREHSOLD_GOAL:
            self.success = SUCCESS

        self.reward = self.reward_A(self.success, self.action_lin, self.action_ang, distance, angle_diff, self.obstacle_distance)

        if self.success != UNKNOWN:
            self.done = True
            if self.success == SUCCESS:
                rospy.loginfo("Goal reached")
            elif self.success == COLLISION:
                rospy.loginfo("Collision detected")
                self.reset_proxy()

        # self.distance_reward = (distance_prev - distance)*(5/time_duration)*1.2*7
        # self.laser_reward = sum(normalized_laser)-24
        # if self.obstacle_distance < THRESHOLD_COLLISION:
        #     self.success = COLLISION
        #     self.collision_reward = -200.0
        # elif self.obstacle_distance < 2*THRESHOLD_COLLISION:
        #     self.collision_reward = -80.0

        # if action[1] > 0.8 or action[1] < -0.8:
        #     self.vel_reward = -1
        # if action[0] < 0.2:
        #     self.vel_reward -= 2
        
        # self.reward = self.distance_reward + self.laser_reward + self.collision_reward + self.vel_reward
            
        return state, self.reward, self.done, self.success
    
if __name__ == '__main__':
    rospy.init_node('drlhf_env')
    env = DRLHFEnv()
    rospy.spin()
        
        