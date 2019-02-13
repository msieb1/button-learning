import pybullet as p
import time


p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)

sphereRadius = 0.05
print(sphereRadius*2)
colBoxId = p.createCollisionShape(p.GEOM_CYLINDER,radius=0.1,height=0.2)
colBoxId2 = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius,sphereRadius,sphereRadius])
print(colBoxId)
mass = 10000
visualShapeId = -1

link_Masses=[1]
linkCollisionShapeIndices=[colBoxId2]
linkVisualShapeIndices=[colBoxId2]
linkPositions=[[0,0,.15]]
linkOrientations=[[0,0,0,1]]
linkInertialFramePositions=[[0,0,0]]
linkInertialFrameOrientations=[[0,0,0,1]]
indices=[0]
jointTypes=[p.JOINT_PRISMATIC]
axis=[[0,0,1]]


basePosition = [0, 0, 0]
baseOrientation = [0,0,0,1]
sphereUid = p.createMultiBody(mass,colBoxId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,
linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,
linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions,
linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)

#p.changeDynamics(sphereUid,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=1.0)
#p.changeDynamics(sphereUid, 1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=10.0)



p.setGravity(0,0,-10)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range (p.getNumJoints(sphereUid)):
	p.getJointInfo(sphereUid,i)
	print(p.getJointInfo(sphereUid,i))
p.enableJointForceTorqueSensor(sphereUid, 0, enableSensor=True)
p.changeDynamics(sphereUid, 1, contactStiffness=10, contactDamping=10)
for joint in range (p.getNumJoints(sphereUid)):
	p.setJointMotorControl2(sphereUid,joint,p.POSITION_CONTROL,targetVelocity=0,force=100)
while (1):


	#print(p.getJointState(sphereUid, 0)[-6:-3])
	time.sleep(0.01)
