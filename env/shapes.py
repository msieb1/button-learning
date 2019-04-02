from abc import ABCMeta, abstractmethod
import pybullet as p
import numpy as np

class SimpleShape(object):
    """Geometric Shape Super Class

    Attributes:

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, params):
        self._params = params
        self._objectId = None

    @abstractmethod
    def set_params(self, params):
        """"Dynamically change shape parameters"""
        self._params = params
        pass

    @property
    @abstractmethod
    def params(self):
        return self._params

    @property
    @abstractmethod
    def objectId(self):
        return self._objectId

class SimpleComposedShape(SimpleShape):
    """Geometric Composed Shape Super Class

    Attributes:

    """
    __metaclass__ = ABCMeta


class Cube(SimpleShape):

    def __init__(self, params):
        assert len(params) == 3
        self.param_dims = len(params)
        self._params = params
        self._objectId = p.createCollisionShape(p.GEOM_BOX,halfExtents=params)
        # p.changeVisualShape(self._objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])

class Sphere(SimpleShape):

    def __init__(self, params):
        assert len(params) == 1
        self._params = params
        self.objectId = p.createCollisionShape(p.GEOM_SPHERE,radius=params)
        # p.changeVisualShape(self._objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])

class Cylinder(SimpleShape):

    def __init__(self, params):
        assert len(params) == 2
        self.param_dims = len(params)
        self._params = params
        self._objectId = p.createCollisionShape(p.GEOM_CYLINDER,radius=params[0], height=params[1])
        # p.changeVisualShape(self._objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])

class Prism(SimpleShape):

    def __init__(self, params):
        assert len(params) == 3
        self.param_dims = len(params) 
        self._params = params
        self._objectId = p.createCollisionShape(p.GEOM_MESH,fileName="prism.obj", collisionFrameOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0, 0]) ,meshScale=params)
        # p.changeVisualShape(self._objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])

class Button(SimpleComposedShape):

    def __init__(self, base_shape, link1_shape, params):
        self._params = params

        # p.createCollisionShape(p.GEOM_PLANE)
        # p.createMultiBody(0,0)
        # self._baseId = globals()[base_shape](params['base_params']).objectId     # calls class shape dynamically
        # self._link1Id = globals()[link1_shape](params['link1_params']).objectId

        # # link masses etc.
        # mass = 10000
        # visualShapeId = -1
        # link_Masses=[1]
        # linkCollisionShapeIndices=[self._link1Id]
        # linkVisualShapeIndices=[self._link1Id]
        # linkPositions=[[0,0,.13]]
        # linkOrientations=[[0,0,0,1]]
        # linkInertialFramePositions=[[0,0,0]]
        # linkInertialFrameOrientations=[[0,0,0,1]]
        # indices=[0]
        # jointTypes=[p.JOINT_PRISMATIC]
        # axis=[[0,0,1]]

        # # Positioning
        # basePosition = [0, 0, 0]
        # baseOrientation = [0,0,0,1]
        # self._objectId = p.createMultiBody(mass, self._baseId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,
        #                             linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,
        #                             linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions,
        #                             linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)

        # # Change joint/body dynamics
        # p.changeDynamics(self.objectId,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=1.0)
        # p.changeDynamics(self.objectId, 1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=10.0)

        # p.getNumJoints(self.objectId)
        # for i in range (p.getNumJoints(self.objectId)):
        #     p.getJointInfo(self.objectId,i)
        
        # # enable force feedback
        # p.enableJointForceTorqueSensor(self.objectId, 0, enableSensor=True)
        # p.changeDynamics(self.objectId, 1, contactStiffness=10, contactDamping=10)
        # for joint in range (p.getNumJoints(self.objectId)):
        #     p.setJointMotorControl2(self.objectId,joint,p.POSITION_CONTROL,targetVelocity=0,force=200)
        
        
        
        # p.createCollisionShape(p.GEOM_PLANE)
        # p.createMultiBody(0,0)
        baseId = globals()[base_shape](params['base_params']).objectId
        link1Id = globals()[link1_shape](params['link1_params']).objectId

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
        self._objectId = p.createMultiBody(mass,baseId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,
                        linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,
                        linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions,
                        linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)
        if self.params['joint_type']==p.JOINT_PRISMATIC:
            p.changeVisualShape(self._objectId,-1,rgbaColor=[0.,1.,1.,1.])
        else:
            p.changeVisualShape(self._objectId,-1,rgbaColor=np.random.rand(3).tolist() + [1])
            p.changeVisualShape(self._objectId, 0,rgbaColor=np.random.rand(3).tolist() + [1])

           

        p.changeDynamics(self._objectId,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=1.0)
        p.changeDynamics(self._objectId, 1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=10.0)

        p.getNumJoints(self._objectId)
        for i in range (p.getNumJoints(self._objectId)):
            p.getJointInfo(self._objectId,i)
        p.enableJointForceTorqueSensor(self._objectId, 0, enableSensor=True)
        p.changeDynamics(self._objectId, 1, contactStiffness=10, contactDamping=10)
        for joint in range (p.getNumJoints(self._objectId)):
            p.setJointMotorControl2(self._objectId,joint,p.POSITION_CONTROL,targetPosition=0,force=200)


# class Button(object):

#     def __init__(self, base_shape, link1_shape, params):
#         p.createCollisionShape(p.GEOM_PLANE)
#         p.createMultiBody(0,0)
#         self._baseId = globals()[base_shape](params['base_params']).objectId     # calls class shape dynamically
#         self._link1Id = globals()[link1_shape](params['link1_params']).objectId

#         # link masses etc.
#         mass = 10000
#         visualShapeId = -1
#         link_Masses=[1]
#         linkCollisionShapeIndices=[self._link1Id]
#         linkVisualShapeIndices=[self._link1Id]
#         linkPositions=[[0,0,.13]]
#         linkOrientations=[[0,0,0,1]]
#         linkInertialFramePositions=[[0,0,0]]
#         linkInertialFrameOrientations=[[0,0,0,1]]
#         indices=[0]
#         jointTypes=[p.JOINT_PRISMATIC]
#         axis=[[0,0,1]]

#         # Positioning
#         basePosition = [0, 0, 0]
#         baseOrientation = [0,0,0,1]
#         self._objectId = p.createMultiBody(mass, self._baseId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,
#                                     linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,
#                                     linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions,
#                                     linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)

#         # Change joint/body dynamics
#         p.changeDynamics(self.objectId,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=1.0)
#         p.changeDynamics(self.objectId, 1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=10.0)

#         p.getNumJoints(self.objectId)
#         for i in range (p.getNumJoints(self.objectId)):
#             p.getJointInfo(self.objectId,i)
        
#         # enable force feedback
#         p.enableJointForceTorqueSensor(self.objectId, 0, enableSensor=True)
#         p.changeDynamics(self.objectId, 1, contactStiffness=10, contactDamping=10)
#         for joint in range (p.getNumJoints(self.objectId)):
#             p.setJointMotorControl2(self.objectId,joint,p.POSITION_CONTROL,targetVelocity=0,force=200)
