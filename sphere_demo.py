from time import sleep
import pybullet as p
import numpy as np
import math

from utils.util import create_ranges_divak

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
                connector = LineTrajectoryGenerator(connector_T, last_traj_end, cur_traj_start)
                self.composed = np.concatenate([self.composed, connector, trajectories[i]])            

        


class PrismaticController(object):
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
            p.resetJointState(sphereId,i,resetJointPoses[i])
        
        
    
    

def setUpWorld(initialSimSteps=100):
    """
    Reset the simulation to the beginning and reload all models.

    Parameters
    ----------
    initialSimSteps : int

    Returns
    -------
    sphereId : int
    endEffectorId : int 
    """
    p.resetSimulation()

    
    # Load plane
    p.loadURDF("plane.urdf", [0, 0, -0.0], useFixedBase=True)

    sleep(0.1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    # Load Baxter
    sphereId = p.loadURDF("prismatic_sphere.urdf", useFixedBase=True)
    buttonId = p.loadURDF("button.urdf", useFixedBase=False)

    p.resetBasePositionAndOrientation(sphereId, [1.0, 0, 0.6], [0., 0., 0., 1.])
    p.resetBasePositionAndOrientation(buttonId, [0, 0, 0.1], [0., 0., 0., 1.])

    #p.resetBasePositionAndOrientation(sphereId, [0.5, -0.8, 0.0],[0,0,0,1])
    #p.resetBasePositionAndOrientation(sphereId, [0, 0, 0], )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    # Grab relevant joint IDs

    # Set gravity
    # p.setGravity(0., 0., -10)

    # Let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return sphereId, buttonId

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

def accurateIK(bodyId, endEffectorId, targetPosition, lowerLimits, upperLimits, jointRanges, restPoses, 
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float] 
    upperLimits : [float] 
    jointRanges : [float] 
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float

    Returns
    -------
    jointPoses : [float] * numDofs
    """
    closeEnough = False
    iter = 0
    dist2 = 1e30

    numJoints = p.getNumJoints(sphereId)

    while (not closeEnough and iter<maxIter):
        jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition)
    
        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(bodyId,i,jointPoses[qIndex-7])
        ls = p.getLinkState(bodyId,endEffectorId)    
        newPos = ls[4]
        diff = [targetPosition[0]-newPos[0],targetPosition[1]-newPos[1],targetPosition[2]-newPos[2]]
        dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
        print("dist2=",dist2)
        closeEnough = (dist2 < threshold)
        iter=iter+1
    print("iter=",iter)
    return jointPoses

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
                                    targetPosition=jointPoses[qIndex-7])



if __name__ == "__main__":
    guiClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(2., 180, 0., [0.52, 0.2, np.pi/4.])


    targetPosXId = p.addUserDebugParameter("targetPosX",-1,1,0)
    targetPosYId = p.addUserDebugParameter("targetPosY",-1,1,0)
    targetPosZId = p.addUserDebugParameter("targetPosZ",-1,1,-0)
    # nullSpaceId = p.addUserDebugParameter("nullSpace",0,1,1)

    sphereId, buttonId = setUpWorld()

    sphere_robot = PrismaticController(sphereId)

    # lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(sphereId, includeFixed=False)

    
    #targetPosition = [0.2, 0.8, -0.1]
    #targetPosition = [0.8, 0.2, -0.1]
    targetPosition = [0.0, 0.0, -0.8]
    
    p.addUserDebugText("TARGET", targetPosition, textColorRGB=[1,0,0], textSize=1.5)

    p.setRealTimeSimulation(0)
    
    maxIters = 10000000

    sleep(1.)
    lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(sphereId, includeFixed=False)

    p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL )

    segments = []
    T = 70
    
    r = 1.0

    

    for _ in range(10):
        theta = np.random.uniform(-math.pi, math.pi)
        
        x_button = np.random.uniform(-1., 1.)
        y_button = np.random.uniform(-1., 1.)
        orn_button = np.random.uniform(-3, 3)
        p.resetBasePositionAndOrientation(buttonId, [x_button, y_button, 0.1], [0., 0., orn_button, 1.])

        x_start = np.cos(theta)
        x_end = 0 + x_button
        y_start = np.sin(theta)
        y_end = 0 + y_button
        z_start = 0.6
        z_end = np.random.uniform(0.4, 0.8)

        # Reset robot
        target_joint_pos = [0.0, 0, 0]
        sphere_robot.reset([x_start - 1.0, y_start, z_start - 0.6])

        segment_1 = LineTrajectoryGenerator(T, [x_start, y_start, z_start], [x_end, y_end, z_end])

        x_start = x_end
        x_end = x_start
        y_start = y_end
        y_end = y_start
        z_start = z_end
        z_end = 0
        segment_2 = LineTrajectoryGenerator(T, [x_start, y_start, z_start], [x_end, y_end, z_end])

        composed = TrajectoryComposer([segment_1.trajectory, segment_2.trajectory])


        for t in range(len(composed.composed)):
            targetPosition = composed.composed[t, ...]
            p.stepSimulation()
            targetPosX = targetPosition[0]
            targetPosY = targetPosition[1]
            targetPosZ = targetPosition[2]

            targetPosition=[targetPosX,targetPosY,targetPosZ]

            sphere_robot.move_to_pos(targetPosition)
            sleep(0.01)


    
        # Reset button
        target_joint_pos = [0.0, 0, 0.1]
        for i in range (p.getNumJoints(buttonId)):
            p.resetJointState(buttonId,i,target_joint_pos[i])
