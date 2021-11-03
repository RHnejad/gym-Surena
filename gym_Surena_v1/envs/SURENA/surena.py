import pybullet as p
import time
import pybullet_data



physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally


p.setGravity(0,0,-9.81)
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
planeId = p.loadURDF("plane.urdf")
modelID=p.loadURDF("sfixed.urdf",startPos, startOrientation)


for i in range (100): #10000
    p.stepSimulation()
    time.sleep(1./240.)
    SPos, SOrn = p.getBasePositionAndOrientation(modelID)
    #print( SOrn)


print("----")
cubePos, cubeOrn = p.getBasePositionAndOrientation(modelID)


print("----")
joints=[i for i in range(28)]
states=p.getLinkStates(modelID,joints)
print(states[7])

print("*")
print(p.getJointStates(modelID,joints))


print("----")
print(p.getBodyInfo(modelID))
print(p.getNumJoints(modelID))
print(p.getJointInfo(modelID,0))


print("----")
print(p.getConnectionInfo())
