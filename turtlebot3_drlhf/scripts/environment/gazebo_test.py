from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import numpy as np

rospy.init_node('test')
spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

entity_path = '/home/face/drlhf_ws/src/turtlebot3_drl/turtlebot3_drlhf/models/turtlebot3_drl_world/goal_box/model.sdf'
entity = open(entity_path, 'r').read()
   

''' Spawn the goal point model'''
goal_pose = Pose()
goal_pose.position.x = np.random.uniform(-2.2, 2.2)
goal_pose.position.y = np.random.uniform(-2.2, 2.2)
goal_pose.position.z = 0.0


def spawn_goal_point():
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        resp = spawn_model('goal_circle', entity, '', goal_pose, 'world')
        if resp.success:
            print('Goal point model spawned')
    except rospy.ServiceException as e:
        rospy.logerr('Spawn Model service call failed')

def delete_goal_point():
    ''' Remove the goal point entity'''
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        resp = delete_model('goal_circle')
        if resp.success:
            print('Goal point model removed')
    except rospy.ServiceException as e:
        rospy.logerr('Delete Model service call failed')

def pause_sim():
    ''' Pause the physics'''
    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause_physics()
        print('Physics paused')
    except rospy.ServiceException as e:
        rospy.logerr('Pause Physics service call failed')

def unpause_sim():
    ''' Unpause the physics'''
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause_physics()
        print('Physics unpaused')
    except rospy.ServiceException as e:
        rospy.logerr('Unpause Physics service call failed')

def reset_sim():
    ''' reset the simulation '''
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        reset_simulation()
        print('Simulation reset')
    except rospy.ServiceException as e:
        rospy.logerr('Reset Simulation service call failed')

def set_ms():
    ''' Set the model state'''
    model_state = ModelState()
    model_state.model_name = 'goal_circle'
    model_state.pose.position.x = np.random.uniform(-2.2, 2.2)
    model_state.pose.position.y = np.random.uniform(-2.2, 2.2)
    model_state.pose.position.z = 0.0

    rospy.wait_for_service('/gazebo/set_model_state')

    try:
        set_model_state(model_state)
        print('Model state set')
    except rospy.ServiceException as e:
        rospy.logerr('Set Model State service call failed')

def model_exists(model_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp = get_model_state(model_name, '')  # Second parameter is for the relative entity name, can be left empty

        if resp.success:
            rospy.loginfo("Model [%s] exists in Gazebo." % model_name)
            return True
        else:
            rospy.loginfo("Model [%s] does not exist in Gazebo." % model_name)
            return False
    except rospy.ServiceException as e:
        rospy.logerror("Service call failed: %s" % e)
        return False

def change_static_model_pose(model_name, new_pose, model_sdf):
    # Wait for the delete_model service to be available
    if model_exists(model_name):
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_resp = delete_model(model_name)
            rospy.loginfo(delete_resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("Delete Model service call failed: %s" % e)

    # Wait for the spawn_sdf_model service to be available
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_resp = spawn_model(model_name, model_sdf, "", new_pose, "world")
        rospy.loginfo(spawn_resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn SDF Model service call failed: %s" % e)
    
    rospy.sleep(1)
    if not model_exists(model_name):
        rospy.logerr("Model [%s] was not successfully spawned." % model_name)


list_goals = [[2.01, -2.04],[0.94, -1.63],[1.55, 0.94],[2.01, 2.04],[-1.8, 0.708],\
              [-1.5, -0.43],[1.15, -2.03],[-2.03, -2.13],[2.18, 1.4],[-2.11, 0.003],\
              [0.49, 1.35],[2.08, -0.244],[0.8, 0.8],[-0.7, -1.06],[-0.71, 2.07],\
              [-2.07, 2.07],[-0.66, -1.68], [-1.25,1.1],[-0.04,2.09],[-0.18,0.54]]

step = 7

if step == 0:
    spawn_goal_point()
elif step == 1:
    delete_goal_point()
elif step == 2:
    pause_sim()
elif step == 3:
    unpause_sim()
elif step == 4:
    reset_sim()
elif step == 5: 
    set_ms()
elif step == 6:
    reps = model_exists('goal_circle')
    print(reps)
elif step == 7:
    new_pose = Pose()
    index = np.random.randint(0, len(list_goals))
    goal = list_goals[index]
    new_pose.position.x = goal[0]
    new_pose.position.y = goal[1]
    print(new_pose)
    change_static_model_pose('goal_circle', new_pose, entity)




