import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 1000
MODE = p.GUI # p.GUI or p.DIRECT - with or without rendering
DIM_OBS = 9 # no. of dimensions in observation space
DIM_ACT = 7 # no. of dimensions in action space

class KukaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(MODE)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*DIM_ACT), np.array([1]*DIM_ACT))
        self.observation_space = spaces.Box(np.array([-1]*DIM_OBS), np.array([1]*DIM_OBS))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        #=========================================================================#
        #  Execute Actions                                                        #
        #=========================================================================#

        # METHOD 1: action = dx,dy,dz in cartesian space
        # orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        # dv = 0.05 # default: 0.005, how big are the actions
        # dx = action[0] * dv
        # dy = action[1] * dv
        # dz = action[2] * dv
        # fingers = action[3]

        # currentPose = p.getLinkState(self.kukaUid, 11)
        # currentPosition = currentPose[0]
        # newPosition = [currentPosition[0] + dx,
        #                currentPosition[1] + dy,
        #                currentPosition[2] + dz]
        # jointPoses = p.calculateInverseKinematics(self.kukaUid,11,newPosition, orientation)[0:7]

        # p.setJointMotorControlArray(self.kukaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        # METHOD 2: action = delta_q
        # Get the current joint angles
        joint_angles = [p.getJointState(self.kukaUid, i)[0] for i in range(7)]

        # Apply the delta_q values from the action vector
        dv = 0.1 # how big are the actions
        joint_angles = [a + action[i]*dv for i, a in enumerate(joint_angles)]

        p.setJointMotorControlArray(self.kukaUid, list(range(7)), p.POSITION_CONTROL, joint_angles)
        
        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.kukaUid, 13)[0] + p.getLinkState(self.kukaUid, 10)[0]
        
        # Kuka arm tip state
        tip_state = p.getLinkState(self.kukaUid, 13)[0] #p.getJointState(self.kukaUid, 10)

        #=========================================================================#
        #  Reward Function and Episode End States                                 #
        #=========================================================================#
        
        # Dense Reward:
        done = False
        tip = tip_state
        obj = state_object
        result = [abs(tip[i] - obj[i]) for i in range(len(tip))]
        reward = -sum(result)
        if state_object[2]>0.45:
            reward += 1
            done = True
        
        # End episode 
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN or reward > -0.09:
            #reward = 0
            done = True
        # Detect Collisions of Kuka and Object
        contacts = p.getContactPoints(self.kukaUid, self.objectUid)
        if contacts:
            print("Collision detected!")
            done = True
        
        #print("REWARD: ",reward)
        info = {'object_position': state_object}
        self.observation = state_robot + state_object # joint_states
        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.connect(MODE) # for testing learned policy must be p.GUI!
        p.resetSimulation()
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)
        
        #=========================================================================#
        #  Generate plane, robotic arm, table and target object                     #
        #=========================================================================#
        # Import a plane
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        
        # Import robotic arm kuka
        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.kukaUid = p.loadSDF(os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper.sdf"))[0]

        for i in range(9):
            p.resetJointState(self.kukaUid,i, rest_poses[i])
        
        # Import a table
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        
        # target object
        state_object= (random.uniform(0.6,0.8),random.uniform(-0.2,0.2),0.05)
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "lego/lego.urdf"), basePosition=state_object)
        state_object = p.getBasePositionAndOrientation(self.objectUid)[0]

        #=========================================================================#
        #  Generate obstacles                                                     #
        #=========================================================================#
        #state_obstacle1 = (random.uniform(0.3,0.9),random.uniform(-0.3, 0.3),0.05)
        
        # OPTION 1: obstacle in front of the target object
        state_obstacle1 = (state_object[0]-0.2, state_object[1],0.05)
        self.obstacle1 = p.loadURDF("gym_panda/envs/cylinder.urdf", basePosition=state_obstacle1)
        
        # OPTION 2: generate X number of obstacles in random positions
        # for i in range(3):
        #     state_obstacle = (random.uniform(0.3,0.9),random.uniform(-0.3, 0.3),0.05)
        #     obstacle_name = 'obstacle' + str(i)
        #     setattr(self, obstacle_name, p.loadURDF("gym_panda/envs/cylinder.urdf", basePosition=state_obstacle))
        
        #=========================================================================#
        #  Observation definition                                                 #
        #=========================================================================#
        state_robot = p.getLinkState(self.kukaUid, 13)[0] + p.getLinkState(self.kukaUid, 10)[0]
        self.observation = state_robot + state_object # joint_states
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
