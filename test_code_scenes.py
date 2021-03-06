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
random.seed(0)

from utils.util import create_ranges_divak
from utils.io_utils import create_data_folders

#dims_dict = {'Cube': 3, 'Cylinder': 2, 'Prism': 3}
dims_dict = {'Cube': 3, 'Cylinder': 2}
object_names = list(dims_dict.keys())

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

    def save_image_data(self, t, results, identifier=''):  
        """
        Save images to file (RGB, Depth, Optical Flow, Mask.png, Mask.npy)
        """

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

    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.params = [yaml.load(f)['camera_0']]
        self._setup_cameras()  

    def _setup_cameras(self):
        self.viewMatrices = []
        self.projectionMatrices = []
        for i in range(len(self.params)):

            self.viewMatrices.append(p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.params[i]['view_params']['camera_target_position'],
                distance=self.params[i]['view_params']['distance'], 
                yaw=self.params[i]['view_params']['yaw'], 
                pitch=self.params[i]['view_params']['pitch'], 
                roll=self.params[i]['view_params']['roll'], 
                upAxisIndex=self.params[i]['view_params']['upAxisIndex']
            ))
        for i in range(len(self.viewMatrices)):
            self.projectionMatrices.append(
            p.computeProjectionMatrixFOV(self.params[i]['proj_params']['fov'], 
                self.params[i]['proj_params']['aspect'], self.params[i]['proj_params']['nearPlane'], self.params[i]['proj_params']['farPlane']))
   
    def get_camera_results(self, view=0):
        img_as_array = p.getCameraImage(self.params[view]['img_width'], self.params[view]['img_height'], self.viewMatrices[view],self.projectionMatrices[view], shadow=0,lightDirection=[1,1,1],renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(np.array(img_as_array[2]), (img_as_array[0], img_as_array[1], 4))
        depth = np.reshape(np.array(img_as_array[3]), (img_as_array[0], img_as_array[1]))
        segmentation = np.reshape(np.array(img_as_array[4]), (img_as_array[0], img_as_array[1]))
        return (img_as_array[0], img_as_array[1], rgb, depth, segmentation) 


class Cube(object):

    def __init__(self, dimensions):
        assert len(dimensions) == 3
        self.dimensions = dimensions
        self.objectId = p.createCollisionShape(p.GEOM_BOX,halfExtents=dimensions)

    def get_objectId(self):
        return self.objectId

# class Sphere(object):

#     def __init__(self, dimensions):
#         assert len(dimensions) == 1
#         self.dimensions = dimensions
#         self.objectId = p.createCollisionShape(p.GEOM_SPHERE,radius=dimensions)

#     def get_objectId(self):
#         return self.objectId

#     def set_dimensions(self, dimensions):
#         pass

class Cylinder(object):

    def __init__(self, dimensions):
        assert len(dimensions) == 2
        self.dimensions = dimensions
        self.objectId = p.createCollisionShape(p.GEOM_CYLINDER,radius=dimensions[0], height=dimensions[1])

    def get_objectId(self):
        return self.objectId

    def set_dimensions(self, dimensions):
        pass

class Prism(object):

    def __init__(self, dimensions):
        assert len(dimensions) == 3
        self.dimensions = dimensions
        # Prism Dimension (0.1 m)
        self.objectId = p.createCollisionShape(p.GEOM_MESH,fileName="prism.obj", collisionFrameOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0, 0]) ,meshScale=dimensions)
        #self.objectId = p.createCollisionShape(p.GEOM_MESH,fileName="prism.obj", collisionFrameOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0, 0]))

    def get_objectId(self):
        return self.objectId

    def set_dimensions(self, dimensions):
        pass

class Button():

    def __init__(self, base_shape, link1_shape, params):
        p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0,0)
        baseId = globals()[base_shape](params['base_params']).get_objectId()
        link1Id = globals()[link1_shape](params['link1_params']).get_objectId()

        mass = 10000
        visualShapeId = -1

        link_Masses=[1]
        linkCollisionShapeIndices=[link1Id]
        linkVisualShapeIndices=[-1]
        linkPositions=[params['link1_pos']]
        linkOrientations=[params['link1_orn']]
        linkInertialFramePositions=[[0,0,0]]
        linkInertialFrameOrientations=[[0,0,0,1]]
        indices=[0]
        jointTypes=[params['joint_type']]
        axis=[[0,0,1]]

        basePosition = params['base_pos']
        baseOrientation = params['base_orn']
        self.objectId = p.createMultiBody(mass,baseId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)
        if params['joint_type']==p.JOINT_PRISMATIC:
            p.changeVisualShape(self.objectId,-1,rgbaColor=[0.,1.,1.,1.])
        else:
            p.changeVisualShape(self.objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])
            p.changeVisualShape(self.objectId, 0,rgbaColor=np.random.rand(3).tolist() + [1])

           

        p.changeDynamics(self.objectId,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=1.0)
        p.changeDynamics(self.objectId, 1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=10.0)

        p.getNumJoints(self.objectId)
        for i in range (p.getNumJoints(self.objectId)):
            p.getJointInfo(self.objectId,i)
        p.enableJointForceTorqueSensor(self.objectId, 0, enableSensor=True)
        p.changeDynamics(self.objectId, 1, contactStiffness=10, contactDamping=10)
        for joint in range (p.getNumJoints(self.objectId)):
            p.setJointMotorControl2(self.objectId,joint,p.POSITION_CONTROL,targetPosition=0,force=200)

    def getId(self):
        return self.objectId

class LineTrajectoryGenerator(object):
    def __init__(self, T, start_pos, end_pos):
        self.T = T
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.trajectory = create_ranges_divak(np.array(start_pos), np.array(end_pos), T).T


class TrajectoryComposer(object):
    def __init__(self, trajectories, connector_T=10):
        self.trajectories = trajectories
        self.connector_T = connector_T
        try:
            composed = trajectories[0]
        except:
            print("provided invalid trajectories")

        for i in range(len(trajectories) - 1):
            idx = i + 1
            last_traj_end = composed[-1]
            cur_traj_start = trajectories[idx][0]
            if np.allclose(cur_traj_start, last_traj_end):
                composed = np.concatenate([composed, trajectories[idx]])
            else:
                connector = LineTrajectoryGenerator(connector_T, last_traj_end, cur_traj_start)
                composed = np.concatenate([composed, connector, trajectories[idx]])
        self.composed = composed

    def add_trajectory_segments(self, trajectories):

        for i in range(len(trajectories)):
            last_traj_end = self.composed[-1]
            cur_traj_start = trajectories[i][0]
            if cur_traj_start == last_traj_end:
                self.composed = np.concatenate([self.composed, trajectories[i]])
            else:
                connector = LineTrajectoryGenerator(self.connector_T, last_traj_end, cur_traj_start)
                self.composed = np.concatenate([self.composed, connector, trajectories[i]])




class PrismaticRobot(object):
    def __init__(self, bodyId):
        self.endEffectorId = p.getNumJoints(bodyId) - 1
        self.bodyId = bodyId
        self.base_pos = p.getLinkState(bodyId, self.endEffectorId)[0]
        self.base_orn = p.getLinkState(bodyId, self.endEffectorId)[1]

    def _set_joint_controls(self, jointPoses):
        """
        Parameters
        ----------
        jointPoses : [float] * numDofs
        """
        numJoints = p.getNumJoints(self.bodyId)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.bodyId, i)
            #print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.setJointMotorControl2(bodyIndex=self.bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[qIndex-7])

    def move_to_pos(self, pos):
        target_pos = np.array(pos) - np.array(self.base_pos)
        self._set_joint_controls(target_pos)

    def reset(self, resetJointPoses):
        for i in range (p.getNumJoints(self.bodyId)):
            p.resetJointState(robotId,i,resetJointPoses[i])


def createIndices(n_total_objects,n_buttons,n_real_buttons):
    button_idx=list(random.sample(range(n_total_objects),n_buttons))
    object_idx = [x for x in range(n_total_objects) if x not in button_idx]
    real_button_idx=list(random.sample(button_idx,n_real_buttons))
    fake_button_idx = [x for x in button_idx if x not in real_button_idx]
    return real_button_idx,fake_button_idx,object_idx

def createGridPositions(grid_size,grid_elements,n_total_obejcts):
    gridPositions=[]
    grid_el_size=float(grid_size)/grid_elements
    for i in range (grid_elements):
    	for j in range (grid_elements):
    	       gridPositions.append([-1+i*grid_el_size,-1+j*grid_el_size,0])
    grid_n =  random.sample(range(len(gridPositions)),n_total_obejcts)
    return [gridPositions[i] for i in grid_n]

def calculateLinkZPos(base_object_name,link1_object_name,base_params,link1_params,joint_type):
    #if joint_type==p.JOINT_PRISMATIC:
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
        # else:
        # if base_object_name=='Cube':
        #     link_pos_z_dim=base_params[2]*2
        # elif base_object_name=='Cylinder':
        #     link_pos_z_dim=base_params[1]
        # elif base_object_name=='Prism':
        #     link_pos_z_dim=base_params[1]*0.1
    return link_pos_z_dim

def createBaseParams(grid_size,grid_elements,base_object_name):
    if base_object_name=='Cube':
        base_params = np.random.uniform(float(grid_size)/(8*grid_elements), float(grid_size)/(6*grid_elements), dims_dict[base_object_name])
    elif base_object_name=='Cylinder':
        radius=np.random.uniform(float(grid_size)/(8*grid_elements), float(grid_size)/(6*grid_elements))
        height=np.random.uniform(float(grid_size)/(8*grid_elements), float(grid_size)/(4*grid_elements))
        base_params=[radius,height]
    elif base_object_name=='Prism':
        base_params=10*np.random.uniform(float(grid_size)/(8*grid_elements), float(grid_size)/(6*grid_elements),dims_dict[base_object_name])
    return base_params

def createLinkParams(grid_size,grid_elements,link1_object_name):
    if link1_object_name=='Cube':
        link1_params = np.random.uniform(float(grid_size)/(16*grid_elements), float(grid_size)/(12*grid_elements), dims_dict[link1_object_name])
    elif link1_object_name=='Cylinder':
        radius=np.random.uniform(float(grid_size)/(16*grid_elements), float(grid_size)/(12*grid_elements))
        height=np.random.uniform(float(grid_size)/(12*grid_elements), float(grid_size)/(8*grid_elements))
        link1_params=[radius,height]
    elif link1_object_name=='Prism':
        link1_params=10*np.random.uniform(float(grid_size)/(12*grid_elements), float(grid_size)/(8*grid_elements),dims_dict[link1_object_name])
    return link1_params

def createParams(grid_size,grid_elements,gridPosition,joint_type):
    params={}
    base_object_name =  np.random.choice(object_names) # these are strings
    link1_object_name = np.random.choice(object_names) # these are strings
    base_params = createBaseParams(grid_size,grid_elements,base_object_name)
    link1_params= createLinkParams(grid_size,grid_elements,link1_object_name)
    print(base_params)
    print(base_object_name)
    base_pos=gridPosition
    print(base_pos)
    link_pos_z_dim=calculateLinkZPos(base_object_name,link1_object_name,base_params,link1_params,joint_type)
    print(link_pos_z_dim)
    link1_pos=[0,0,link_pos_z_dim]
    #base_orn=[0., 0.,np.random.uniform(0,np.pi), 1.] #TBD
    #link1_orn=[0., 0.,np.random.uniform(np.pi), 1.] #TBD
    base_orn=[0.,0.,0.,1]
    link1_orn=[0.,0.,0.,1]
    params={'base_params': base_params, 'link1_params': link1_params,'base_pos':base_pos,'link1_pos':link1_pos,'base_orn':base_orn,'link1_orn':link1_orn,'joint_type':joint_type,'link_z':link_pos_z_dim}
    return base_object_name,link1_object_name,params

def setUpWorld(grid_size,grid_elements,n_total_obejcts,n_buttons,n_real_buttons):
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

    # Load plane
    p.loadURDF("plane.urdf", [0, 0, -0.0], useFixedBase=True)

    sleep(0.1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    # Load Baxter
    robotId = None
    # robotId = p.loadURDF("prismatic_sphere.urdf", useFixedBase=True)
    # p.enableJointForceTorqueSensor(robotId, 0, enableSensor=True)

    # buttonId = Button(base_shape='Cube', link1_shape='Cube', params={'base_dimensions': [0.1, 0.1, 0.1], 'link1_dimensions': [0.05, 0.05, 0.09]}).get_objectId()
#####
    gridPositions=createGridPositions(grid_size,grid_elements,n_total_objects)
    real_button_idx,fake_button_idx,object_idx=createIndices(n_total_objects,n_buttons,n_real_buttons)

    for i in range(n_total_objects):
        if i in real_button_idx:
            base_object_name,link1_object_name,params=createParams(grid_size, grid_elements,gridPositions[i],p.JOINT_PRISMATIC)
            buttonId = Button(base_object_name, link1_object_name,params).getId()
        elif i in fake_button_idx:
            base_object_name,link1_object_name,params=createParams(grid_size, grid_elements,gridPositions[i],p.JOINT_FIXED)
            buttonId = Button(base_object_name, link1_object_name,params).getId()
        else:
            base_object_name,link1_object_name,params=createParams(grid_size, grid_elements,gridPositions[i],False)
            baseId = globals()[base_object_name](params['base_params']).get_objectId()
            # visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="duck.obj", rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
            # collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="duck_vhacd.obj", collisionFramePosition=shift,meshScale=meshScale)
            # buttonId = p.createMultiBody(baseMass=1,baseVisualShapeIndex=visualShapeId,baseCollisionShapeIndex=-1,basePosition=params['base_pos'],baseOrientation=params['base_orn'])
            dict_orn={'Prism':p.getQuaternionFromEuler([math.pi / 2.0, 0, 0]),'Cube':[0.,0.,0.,1],'Cylinder':[0.,0.,0.,1]}
            p.createMultiBody(1,baseId,-1,params['base_pos'],params['base_orn'])#baseInertialFramePosition=[0.,0.,0.])#baseInertialFrameOrientation=dict_orn[base_object_name])
            #baseMass=1,baseInertialFramePosition=[0,0,0],baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex = visualShapeId, basePosition = [((-rangex/2)+i)*meshScale[0]*2,(-rangey/2+j)*meshScale[1]*2,1], useMaximalCoordinates=True
#####       
            p.changeVisualShape(baseId,-1,rgbaColor=np.random.rand(3).tolist() + [1])
    import ipdb; ipdb.set_trace()

    #buttonId = Button(base_shape='Cylinder', link1_shape='Prism', params={'base_dimensions': [0.1, 0.2], 'link1_dimensions': [0.03, 0.15, 0.03]}).getId()
    # self.objectId = p.loadURDF("button.urdf", useFixedBase=False)

    #p.resetBasePositionAndOrientation(robotId, [1.0, 0, 0.6], [0., 0., 0., 1.])
    # p.resetBasePositionAndOrientation(self.objectId, [0, 0, 0.1], [0., 0., 0., 1.])

    #p.resetBasePositionAndOrientation(robotId, [0.5, -0.8, 0.0],[0,0,0,1])
    #p.resetBasePositionAndOrientation(robotId, [0, 0, 0], )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    # Grab relevant joint IDs

    # Set gravity
    p.setGravity(0., 0., -10)

    # Let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return robotId, buttonId

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


if __name__ == "__main__":
    guiClient = p.connect(p.GUI)
    # guiClient = p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(2., 0, -88., [0., 0, 0])


    targetPosXId = p.addUserDebugParameter("targetPosX",-1,1,0)
    targetPosYId = p.addUserDebugParameter("targetPosY",-1,1,0)
    targetPosZId = p.addUserDebugParameter("targetPosZ",-1,1,-0)
    # nullSpaceId = p.addUserDebugParameter("nullSpace",0,1,1)


    grid_size=2
    grid_elements=4
    n_total_objects=10
    n_buttons=5
    n_real_buttons=3

    robotId, buttonId = setUpWorld(grid_size,grid_elements,n_total_objects,n_buttons,n_real_buttons)
    # robot = PrismaticRobot(robotId)

    # lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(robotId, includeFixed=False)


    #targetPosition = [0.2, 0.8, -0.1]
    #targetPosition = [0.8, 0.2, -0.1]
    targetPosition = [0.0, 0.0, -0.8]
    cam_manager = CameraManager('configs/camera_config.yml')
    collector = DataCollector(save_dir=join('experiments', 'button'), opt_flow=False)


    p.setRealTimeSimulation(0)
    results = cam_manager.get_camera_results()
    collector.save_image_data(0, results)


    # maxIters = 10000000

    # sleep(1.)
    # lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(robotId, includeFixed=False)

    # p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL )

    # segments = []
    # T = 70

    # r = 1.0
    # criterion = _LOSS()
    # agent = Agent()

    # boxes = 

    # logits = agent.predict(boxes, classifier_type='rgb')
    # target_classes = torch.Tensor(buttonIds).float()
    # if USE_CUDA:
    #     target_classes = target_classes.cuda()
    # for _ in range(10):
    #     theta = np.random.uniform(-math.pi, math.pi)
        
    #     x_button = np.random.uniform(-1., 1.)
    #     y_button = np.random.uniform(-1., 1.)
    #     orn_button = np.random.uniform(-3, 3)
    #     p.resetBasePositionAndOrientation(buttonId, [x_button, y_button, 0.1], [0., 0., orn_button, 1.])
    
    #     x_start = np.cos(theta)
    #     x_end = 0 + x_button
    #     y_start = np.sin(theta)
    #     y_end = 0 + y_button
    #     z_start = 0.6
    #     z_end = np.random.uniform(0.4, 0.8)
    
    #     # Reset robot
    #     target_joint_pos = [0.0, 0, 0]
    #     robot.reset([x_start - 1.0, y_start, z_start - 0.6])
    
    #     segment_1 = LineTrajectoryGenerator(T, [x_start, y_start, z_start], [x_end, y_end, z_end])
    
    #     x_start = x_end
    #     x_end = x_start
    #     y_start = y_end
    #     y_end = y_start
    #     z_start = z_end
    #     z_end = 0
    #     segment_2 = LineTrajectoryGenerator(T, [x_start, y_start, z_start], [x_end, y_end, z_end])
    
    #     composed = TrajectoryComposer([segment_1.trajectory, segment_2.trajectory])
    
    
    #     for t in range(len(composed.composed)):
    #         targetPosition = composed.composed[t, ...]
    #         p.stepSimulation()
    #         targetPosX = targetPosition[0]
    #         targetPosY = targetPosition[1]
    #         targetPosZ = targetPosition[2]
    
    #         targetPosition=[targetPosX,targetPosY,targetPosZ]
    #         # print(p.getJointState(buttonId, 0)[-6:-3])
    #         # print(p.getJointState(robotId, 0)[-6:-3])
    
    #         robot.move_to_pos(targetPosition)
    #         sleep(0.01)
    
    
    
    #     # Reset button
    #     target_joint_pos = [0.0, 0, 0.1]
    #     for i in range (p.getNumJoints(buttonId)):
    #         p.resetJointState(buttonId,i,target_joint_pos[i])
    while (1):
    	keys = p.getKeyboardEvents()
    time.sleep(0.1)
