import numpy as np
import imageio
import cv2
import os
from os.path import join
import pybullet as p
import random
import yaml

from utils.io_utils import create_data_folders

class DataCollector(object):
    """
    Collects demonstration

    ToDo: parametrize start and end configuration of objects and gripper
    """
    def __init__(self, save_dir, num=None, opt_flow=False, sample_rate_opt_flow=30):
        self.save_dir = save_dir
        self.num = num
        self.opt_flow = opt_flow
        self.sample_rate_opt_flow = sample_rate_opt_flow

        self.setup_io()

    def _get_seq_number(self):
        """
        Create unique identifier, either provided or using latest sequence number in folder +1
        """

        if self.num:
            demo_num = self.num
        else:
            # If there's no video directory, this is the first sequence.
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            files = [f for f in os.listdir(self.save_dir) if not f.endswith('.npy')]

            if not files:
              demo_num = '0'
            else:
              # Otherwise, get the latest sequence name and increment it.
              seq_names = [i.split('_')[0] for i in files if not i.endswith('.mp4')]
              latest_seq = sorted(map(int, seq_names), reverse=True)[0]
              demo_num = str(latest_seq+1)
        return demo_num

    def _compute_optical_flow(self, prev, cur):
        # flow = cv2.calcOpticalFlowFarneback(skimage.img_as_float32(prev), skimage.img_as_float32(cur), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, 0.5, 1, 3, 15, 3, 5, 1, 0)
        return flow

    def setup_io(self):
        """
        Setup io paths and filenames
        """
        print("saving to ", self.save_dir)
        self.demo_num = self._get_seq_number()

        # self.create_snapshot_experiment()

        if self.opt_flow:
            self.rgb_folder, self.depth_folder, self.seg_folder, self.flow_folder, self.info_folder, self.base_folder, self.vid_folder, self.sensor_folder \
                            = create_data_folders(join(self.save_dir, self.demo_num), visual=True) # Put into config folder?
            self.prev_img = None # For computing optical flow
        else:
            self.rgb_folder, self.depth_folder, self.seg_folder, self.info_folder, self.base_folder, self.vid_folder, self.sensor_folder \
                            = create_data_folders(join(self.save_dir, self.demo_num), visual=False) # Put into config folder?            

        """Creates and returns one view directory per webcam."""

    def save_image_data(self, t, camera_results, identifier=''):  
        """
        Save images to file (RGB, Depth, Optical Flow, Mask.png, Mask.npy)
        """
        results = camera_results
        # Save RGB images
        imageio.imwrite('{0}/{1:05d}{2}.png'.format(
            self.rgb_folder, t, identifier), np.array(results[2]))

        # Save depth image
        near = 0.001
        far = 10
        depth_tiny = far * near / (far - (far - near) * results[3])
        imageio.imwrite('{0}/{1:05d}{2}.png'.format(
            self.depth_folder, t, identifier), depth_tiny)

        # Save mask as png
        imageio.imwrite('{0}/{1:05d}{2}.png'.format(
            self.seg_folder, t, identifier), (255*results[4]).astype(np.uint8))

        # Save mask
        np.save('{0}/{1:05d}{2}.npy'.format(
             self.seg_folder, t, identifier), results[4])

        # Save optical flow if enabled
        if self.opt_flow:
            # Note: we can compute flow online
            cur = cv2.cvtColor(results[2], cv2.COLOR_BGR2GRAY) # TODO Check if BGR or RGB
            # First frame
            # if t // self.sample_rate_opt_flow == 1:
            if t == 0:
                self.prev_img = cur
            flow = self._compute_optical_flow(self.prev_img, cur)
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
            # Visualize optical flow
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/math.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
            imageio.imwrite('{0}/{1:05d}{2}.png'.format(self.flow_folder, t, identifier), rgb)
            self.prev_img = cur

class CameraManager(object):

    def __init__(self, camera_config_path_path):
        with open(camera_config_path_path, 'r') as f:
            self._params = yaml.load(f)
        self._setup_cameras()  
    
    @property
    def params(self):
        return self._params

    def change_camera_target(self, new_camera_target, view=0):
        self.viewMatrices[view] = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=new_camera_target,
                        distance=self._params[view]['view_params']['distance'], 
                        yaw=self._params[view]['view_params']['yaw'], 
                        pitch=self._params[view]['view_params']['pitch'], 
                        roll=self._params[view]['view_params']['roll'], 
                        upAxisIndex=self._params[view]['view_params']['upAxisIndex']
                    )

    def _setup_cameras(self):
        self.viewMatrices = []
        self.projectionMatrices = []
        for i in range(len(self._params)):

            self.viewMatrices.append(p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._params[i]['view_params']['camera_target_position'],
                distance=self._params[i]['view_params']['distance'], 
                yaw=self._params[i]['view_params']['yaw'], 
                pitch=self._params[i]['view_params']['pitch'], 
                roll=self._params[i]['view_params']['roll'], 
                upAxisIndex=self._params[i]['view_params']['upAxisIndex']
            ))
        for i in range(len(self.viewMatrices)):
            self.projectionMatrices.append(
            p.computeProjectionMatrixFOV(self._params[i]['proj_params']['fov'], 
                self._params[i]['proj_params']['aspect'], self._params[i]['proj_params']['nearPlane'], self._params[i]['proj_params']['farPlane']))
    
    def add_camera(self, params):
        self.viewMatrices.append(p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=params['view_params']['camera_target_position'],
            distance=params['view_params']['distance'], 
            yaw=params['view_params']['yaw'], 
            pitch=params['view_params']['pitch'], 
            roll=params['view_params']['roll'], 
            upAxisIndex=params['view_params']['upAxisIndex']
        ))        
        
        self.projectionMatrices.append(
        p.computeProjectionMatrixFOV(params['proj_params']['fov'], 
            params['proj_params']['aspect'], params['proj_params']['nearPlane'], params['proj_params']['farPlane']))
        self._params[len(self._params)] = params
        index = len(self.viewMatrices) - 1
        return index
   
    def get_camera_results(self, view=0):
        img_as_array = p.getCameraImage(self._params[view]['img_width'], self._params[view]['img_height'], self.viewMatrices[view],self.projectionMatrices[view], shadow=0,lightDirection=[1,1,1],renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(np.array(img_as_array[2]), (img_as_array[0], img_as_array[1], 4))
        depth = np.reshape(np.array(img_as_array[3]), (img_as_array[0], img_as_array[1]))
        segmentation = np.reshape(np.array(img_as_array[4]), (img_as_array[0], img_as_array[1]))
        return (img_as_array[0], img_as_array[1], rgb, depth, segmentation) 



def getJointRanges(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)

        if includeFixed or jointInfo[3] > -1:

            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(-2)
            upperLimits.append(2)
            jointRanges.append(2)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses

def setMotors(bodyId, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        #print(jointInfo)
        qIndex = jointInfo[3]
        if qIndex > -1:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[qIndex])