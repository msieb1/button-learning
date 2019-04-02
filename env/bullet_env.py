from time import sleep
import pybullet as p
import numpy as np
import math
from abc import ABCMeta, abstractmethod
import random
import yaml
from os.path import join
import os
import imageio
import random
from importlib import import_module
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy as copy
import argparse
random.seed(0)

from utils.env_utils import DataCollector, CameraManager                    
from utils.util import create_ranges_divak
from shapes import Button


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--collect-data', action='store_true')
FLAGS = parser.parse_args()

class BulletEnv(object):

    def __init__(self, env_config_path, headless=False):

        # Load environment config
        with open(env_config_path, 'r') as f:
            self.env_config = yaml.load(f)
        self.object_dict = self.env_config['object_dict']
        self.object_names = list(self.object_dict.keys())
        self.grid_size = self.env_config['grid_size']
        self.grid_elements = self.env_config['grid_elements']
        self.n_total_objects = self.env_config['n_total_objects']
        self.n_buttons = self.env_config['n_buttons']
        self.n_real_buttons = self.env_config['n_real_buttons']

        self.camera_manager = CameraManager('configs/camera_config.yml')

        if not headless:
            # Run GUI or ...
            guiClient = p.connect(p.GUI)
        else:
            # ... headless
            guiClient = p.connect(p.DIRECT)

        # Place Debug Camera Visualizer
        p.resetDebugVisualizerCamera(2., 0, -88., [0., 0, 0])
        targetPosXId = p.addUserDebugParameter("targetPosX",-1,1,0)
        targetPosYId = p.addUserDebugParameter("targetPosY",-1,1,0)
        targetPosZId = p.addUserDebugParameter("targetPosZ",-1,1,-0)

        self.robotId = self._setUpWorld()
        # robot = PrismaticRobot(robotId)
        targetPosition = [0.0, 0.0, -0.8]

    #### PyBullet Env Setup and Object Spawning ####

    def _setUpWorld(self):
        """
        Reset the simulation to the beginning and reload all models.

        Parameters
        ----------
        initialSimSteps : int

        Returns
        -------
        robotId : int
        endEffectorId : int
        """
        
        initialSimSteps=100
        p.resetSimulation()

        # Load plane (is weird right now - somehow always colored and makes image processing slow)
        # planeId = p.loadURDF("plane.urdf", [0, 0, -0.0], useFixedBase=True)
        # p.changeVisualShape(planeId,-1,rgbaColor=[255,255,255,1])

        sleep(0.1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

        # Load Sphere Robot or Not
        robotId = None
        # robotId = p.loadURDF("prismatic_sphere.urdf", useFixedBase=True)
        # p.enableJointForceTorqueSensor(robotId, 0, enableSensor=True)

        # buttonId = Button(base_shape='Cube', link1_shape='Cube', params={'base_dimensions': [0.1, 0.1, 0.1], 'link1_dimensions': [0.05, 0.05, 0.09]}).objectId
        self.grid_elem_size = float(self.grid_size)/self.grid_elements
        self.gridPositions = self._createGridPositions()

        real_button_indices, fake_button_indices, object_indices = self._createIndices()
        print(real_button_indices, fake_button_indices, object_indices)
        self.grid_to_objectId = edict()
        self.grid_to_camId = edict()
        self.id_to_realButton = edict()
        for i in range(len(self.gridPositions)):
            k = np.random.choice([0, 1, 2], 1)
            if i in real_button_indices:
                base_object_name,link1_object_name,params = self._createParams(self.gridPositions[i],p.JOINT_PRISMATIC)
                id = Button(base_object_name, link1_object_name,params).objectId
                self.id_to_realButton.id = True
            elif i in fake_button_indices:
                base_object_name,link1_object_name,params = self._createParams(self.gridPositions[i],p.JOINT_FIXED)
                id = Button(base_object_name, link1_object_name,params).objectId
                self.id_to_realButton.id = False
            else:
                skip = np.random.rand(1)
                if skip >1:
                    id = None
                else:
                    base_object_name,link1_object_name,params = self._createParams(self.gridPositions[i],False)
                    BaseObject = getattr(import_module('env.shapes'), base_object_name)
                    id = BaseObject(params['base_params']).objectId
                    p.createMultiBody(1,id,-1,params['base_pos'],params['base_orn'])#baseInertialFramePosition=[0.,0.,0.])#baseInertialFrameOrientation=dict_orn[base_object_name])
                    p.changeVisualShape(id,-1,rgbaColor=np.random.rand(3).tolist() + [1])
                    self.id_to_realButton.id = False
            self.grid_to_objectId[str(i)] = id

            # Add camera view for current grid cell
            params = copy(self.camera_manager.params[1])
            params['view_params']['camera_target_position'] = self.gridPositions[i]
            self.grid_to_camId[str(i)] = self.camera_manager.add_camera(params)

        # Render scene
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        # Set gravity (disable without loading plane or objects fall through)
        # p.setGravity(0., 0., -10)

        # Set realtime simulation flag
        p.setRealTimeSimulation(0)

        # Let the world run for a bit
        for _ in range(initialSimSteps):
            p.stepSimulation()

        return robotId

    def _createIndices(self):
        button_idx = list(random.sample(range(len(self.gridPositions)),self.n_buttons))
        object_idx = [x for x in range(len(self.gridPositions)) if x not in button_idx]
        real_button_idx = list(random.sample(button_idx, self.n_real_buttons))
        fake_button_idx = [x for x in button_idx if x not in real_button_idx]
        return real_button_idx, fake_button_idx, object_idx

    def _createGridPositions(self):
        gridPositions=[]
        for j in range (self.grid_elements+1):
            for i in range (self.grid_elements+1):
                gridPositions.append([i*self.grid_elem_size,j*self.grid_elem_size,0])
        # grid_n =  random.sample(range(len(gridPositions)), self.n_total_objects)
        return gridPositions

    def _calculateLinkZPos(self, base_object_name, link1_object_name, base_params, link1_params,joint_type):
        if base_object_name=='Cube' and link1_object_name=='Cube':
            link_pos_z_dim=base_params[2]+link1_params[2]
        elif base_object_name=='Cylinder' and link1_object_name=='Cube':
            link_pos_z_dim=base_params[1]*0.5+link1_params[2]
        elif base_object_name=='Prism' and link1_object_name=='Prism':
            link_pos_z_dim=base_params[1]*0.1*0.5+link1_params[1]*0.1*0.5
        elif base_object_name=='Cylinder' and link1_object_name=='Cylinder':
            link_pos_z_dim=base_params[1]*0.5+link1_params[1]*0.5
        elif base_object_name=='Prism' and link1_object_name=='Cylinder':
            link_pos_z_dim=base_params[1]*0.1+link1_params[1]*0.5
        elif base_object_name=='Prism' and link1_object_name=='Cube':
            link_pos_z_dim=base_params[1]*0.1+link1_params[2]
        elif base_object_name=='Cylinder' and link1_object_name=='Prism':
            link_pos_z_dim=base_params[1]*0.5+link1_params[1]*0.1*0.5
        elif base_object_name=='Cube' and link1_object_name=='Cylinder':
            link_pos_z_dim=base_params[2]+link1_params[1]*0.5
        elif base_object_name=='Cube' and link1_object_name=='Prism':
            link_pos_z_dim=base_params[2]+link1_params[1]*0.1*0.5
        return link_pos_z_dim

    def _createBaseParams(self, base_object_name):
        if base_object_name=='Cube':
            base_params = np.random.uniform(float(self.grid_size)/(8*self.grid_elements), float(self.grid_size)/(6*self.grid_elements), self.object_dict[base_object_name])
        elif base_object_name=='Cylinder':
            radius=np.random.uniform(float(self.grid_size)/(8*self.grid_elements), float(self.grid_size)/(6*self.grid_elements))
            height=np.random.uniform(float(self.grid_size)/(8*self.grid_elements), float(self.grid_size)/(4*self.grid_elements))
            base_params=[radius,height]
        elif base_object_name=='Prism':
            base_params=10*np.random.uniform(float(self.grid_size)/(8*self.grid_elements), float(self.grid_size)/(6*self.grid_elements),self.object_dict[base_object_name])
        return base_params

    def _createLinkParams(self, link1_object_name):
        if link1_object_name=='Cube':
            link1_params = np.random.uniform(float(self.grid_size)/(16*self.grid_elements), float(self.grid_size)/(12*self.grid_elements), self.object_dict[link1_object_name])
        elif link1_object_name=='Cylinder':
            radius=np.random.uniform(float(self.grid_size)/(16*self.grid_elements), float(self.grid_size)/(12*self.grid_elements))
            height=np.random.uniform(float(self.grid_size)/(12*self.grid_elements), float(self.grid_size)/(8*self.grid_elements))
            link1_params=[radius,height]
        elif link1_object_name=='Prism':
            link1_params=10*np.random.uniform(float(self.grid_size)/(12*self.grid_elements), float(self.grid_size)/(8*self.grid_elements),self.object_dict[link1_object_name])
        return link1_params

    def _createParams(self, gridPosition, joint_type):
        params={}
        base_object_name =  np.random.choice(self.object_names) # these are strings
        link1_object_name = np.random.choice(self.object_names) # these are strings
        base_params = self._createBaseParams(base_object_name)
        link1_params= self._createLinkParams(link1_object_name)
        base_pos=gridPosition
        link_pos_z_dim = self._calculateLinkZPos(base_object_name, link1_object_name, base_params,link1_params,joint_type)
        link1_pos=[0, 0, link_pos_z_dim]
        #base_orn=[0., 0.,np.random.uniform(0,np.pi), 1.] #TBD
        #link1_orn=[0., 0.,np.random.uniform(np.pi), 1.] #TBD
        base_orn=[0., 0., 0., 1]
        link1_orn=[0., 0., 0., 1]
        params={'base_params': base_params, 'link1_params': link1_params,'base_pos':base_pos,'link1_pos':link1_pos,'base_orn':base_orn,'link1_orn':link1_orn,'joint_type':joint_type,'link_z':link_pos_z_dim}
        return base_object_name,link1_object_name,params

    #### API Functions ####
    
    def sample_action(self):
        action = np.random.choice(np.arange(len(self.gridPositions)), 1)
        return int(action)

    def reset(self):
        self._setUpWorld()

    def step(self, action):
         # Get global camera results from bullet
        camera_results_0 = self.camera_manager.get_camera_results(view=0)

        # Get cropped images per grid cell
        camera_results_1 = self.camera_manager.get_camera_results(view=self.grid_to_camId[str(action)] )
        rgb = camera_results_1[2]
        depth = camera_results_1[3]

        # Gather step data
        observation = {'rgb': rgb, 'depth': depth, 'shape': None, 'color': None}
        reward = self.get_reward(action)

        # Plot some stuff
        # cv2.imshow('global_rgb', camera_results_0[2])
        cv2.imshow('cropped_rgb', camera_results_1[2])
        cv2.waitKey(1)

        return observation, reward

    def get_reward(self, action):
        if self.grid_to_objectId[str(action)] is None:
            return 0
        else:
            return int(self.grid_to_objectId[str(action)]) 
            
    def get_camera_results(self, view=0):
        return self.camera_manager.get_camera_results(view=view)



if __name__ == "__main__":

    env = BulletEnv(env_config_path='configs/env_config.yml')
    data_collector = DataCollector(save_dir=join('experiments', 'button'), opt_flow=False)
    
    t = 0
    while (1):
        action = env.sample_action()

        print('action: {}'.format(action))
        observation, reward = env.step(action)

        if FLAGS.collect_data:
            # data_collector.save_image_data(t=t, camera_results=env.get_camera_results(view=env.grid_to_camId['8']), identifier='_view{}'.format(action)) 
            data_collector.save_image_data(t=t, camera_results=env.get_camera_results(view=0), identifier='_global') 
            for i in range(len(env.gridPositions)):
                data_collector.save_image_data(t=t, camera_results=env.get_camera_results(view=env.grid_to_camId[str(i)]), identifier='_view{}'.format(str(i))) 
                
        t += 1
        # import ipdb; ipdb.set_trace()
        env.reset()

    # Save test image
    env.get_camera_results()
    import ipdb; ipdb.set_trace()

    while (1):
    	keys = p.getKeyboardEvents()
    time.sleep(0.1)
