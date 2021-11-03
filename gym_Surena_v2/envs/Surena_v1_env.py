import pybullet
import time
import pybullet_data
import numpy as np
from Robot import *
from Ankle import *
from DCM import *

KNEE=[]

phisycsClient = pybullet.connect(pybullet.GUI) #,options= "--opengl2"
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.resetSimulation()
planeID = pybullet.loadURDF("plane.urdf")
pybullet.setGravity(0,0,-9.81)
robotID = pybullet.loadURDF("SURENA/sfixed.urdf",
                         [0.0,0.0,0.0],useFixedBase = 0) #"surena4_1.urdf"
pybullet.setRealTimeSimulation(0)


#print("****", pybullet.getBasePositionAndOrientation(1))

dt = 1/240 ##############################################################################################################3
planner = DCMPlanner(0.7, 1.5, 0.45,dt) #deltaZ,  stepTime,  doubleSupportTime,  dt, stepCount = 6, alpha = 0.5

dx=0.15625

rF =np.array([[0.0,-0.115,0.0],
             [0.25,0.115,0.0],
             [0.5,-0.115,0.0],
             [0.75,0.115,0.0],
             [1.0,-0.115,0.0],
             [1.25,0.115,0.0]])

# rF =np.array([[0.0,0.0,0.0],
#              [dx,0.115,0.0],
#              [3*dx,-0.115,0.0],
#              [5*dx,0.115,0.0],
#              [7*dx,-0.115,0.0],
#              [8*dx,0.0,0.0]])

aa=0.115
bb=0.1

# for i in range(1,6):
#     tempx=rF[i-1][0]+bb*np.sqrt(2)/2
#     tempy=rF[i-1][1]+bb*np.sqrt(2)/2
#     rF[i][0]=tempx+aa*((-1)**i)*np.sqrt(2)/2
#     rF[i][1]=tempx-aa*((-1)**i)*np.sqrt(2)/2

# print("&",rF)

planner.setFoot(rF)
xi_trajectory = planner.getXiTrajectory()
com_0 = np.array([0.0,0.0,planner.deltaZ_])
com_trajectory = planner.getCoMTrajectory(com_0)

anklePlanner = Ankle(planner.tStep_, planner.tDS_, 0.05,dt)


rF =np.array([[0.0,0.115,0.0],
            [0.0,-0.115,0.0],
            [0.25,0.115,0.0],
            [0.5,-0.115,0.0],
            [0.75,0.115,0.0],
            [1.0,-0.115,0.0],
            [1.25,0.115,0.0],
            [1.25,-0.115,0.0]])

# rF =np.array([[0.0,0.115,0.0],
#              [0.0,-0.115,0.0],
#              [dx,0.115,0.0],
#              [3*dx,-0.115,0.0],
#              [5*dx,0.115,0.0],
#              [7*dx,-0.115,0.0],
#              [8*dx,0.115,0.0],
#              [8*dx,-0.115,0.0]])

plt.figure()
plt.plot((rF.T)[0],(rF.T)[1])
plt.show()
# for i in range(2,6):
#     tempx=rF[i-1][0]+bb*np.sqrt(2)/2
#     tempy=rF[i-1][1]+bb*np.sqrt(2)/2
#     rF[i][0]=tempx-aa*((-1)**i)*np.sqrt(2)/2
#     rF[i][1]=tempx+aa*((-1)**i)*np.sqrt(2)/2

# rF[7][0]=rF[7][0]
# rF[7][1]=rF[7][1]*-1

# print("$$",rF)

anklePlanner.updateFoot(rF)
anklePlanner.generateTrajectory()
left = np.array(anklePlanner.getTrajectoryL())
right = np.array(anklePlanner.getTrajectoryR())


###################################################
num_actions=12
observation_dimensions=31
jointIDs=[0,1,2,3,4,5,6,7,8,9,10,11]
def Observations(SurenaID,planeId): 

    Ts=np.zeros(num_actions)
    Theta_dots=np.zeros(num_actions)
    Powers=np.zeros(num_actions)

    
    SPos, SOrn = pybullet.getBasePositionAndOrientation(SurenaID)
    LinearVel,AngularVel=pybullet.getBaseVelocity(SurenaID)
    JointStates=pybullet.getJointStates(SurenaID,jointIDs) ##JPos,JVel,JF
    contacts=pybullet.getContactPoints(SurenaID,planeId)
    iscontact=bool(len(contacts))

    link_states=pybullet.getLinkStates(SurenaID,[5,11])
    z_l5,z_l11=link_states[0][0][2],link_states[1][0][2]

    if z_l5>=0.04 and z_l11>=0.04 :
        #print("z_l5: ",z_l5,"z_l11:",z_l11)
        iscontact=False
    elif z_l5<=foot_z0 or z_l11<=foot_z0:
        iscontact=True

    observation_new=np.zeros(observation_dimensions)
    for ii in range(num_actions):
        observation_new[ii]=JointStates[ii][0] #theta
        observation_new[ii+num_actions]=JointStates[ii][1] #theta_dot
        
    FzR,FzL=0.0,0.0
    ncontact=len(contacts)
    for k in range(ncontact):
        if contacts[k][3]==11:
            FzR+=contacts[k][9]
        elif contacts[k][3]==5:
            FzL+=contacts[k][9]


    #FIX FOR 12 DOF
    x=SPos[0]
    currentPos=observation_new[0: num_actions]    
    observation_new[2*num_actions : 2*num_actions+2]=np.array(SPos)[1:3]
    observation_new[2*num_actions+2 : 2*num_actions+5]=np.array(LinearVel)
    observation_new[2*num_actions+5 : 2*num_actions+7]=np.array([FzR, FzL]) #F_z_r and F_z_l
    #observation_new[26:28]=np.array([JointStates[3][2][2] , JointStates[8][2][2]]) #F_z_r and F_z_l


    for jj in range( num_actions):
        Ts[jj]=JointStates[jj][3]
        Theta_dots[jj]=JointStates[jj][1]
        
    Ts=np.absolute(Ts)
    Theta_dots=np.absolute(Theta_dots)    
    powers=sum(Ts*Theta_dots)
    
       

    return observation_new, iscontact, powers, x

#####

w0=5
w1,w2=8,8
w3,w4,w5=0.025,40,40
    
X0=-0.517
Z0=0.9727
foot_z0=0.03799

surena = Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)
OBS=[]
Dtheta=[]
step_counter=0
up=0
time.sleep(1)
for i in range(240):
        All = surena.doIK([0.0,0.0,0.74 - (i/240)* (0.74-planner.deltaZ_)], np.eye(3),[0.0,0.115,0.0], np.eye(3),[0.0, -0.115,0.0], np.eye(3))
        leftConfig = All[6:12]
        rightConfig = All[0:6]
        for index in range (6):
            pybullet.setJointMotorControl2(bodyIndex=robotID,
                                    jointIndex=index,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPosition = rightConfig[index])
            pybullet.setJointMotorControl2(bodyIndex=robotID,
                                    jointIndex=index + 6,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPosition = leftConfig[index])
        pybullet.stepSimulation()
        KNEE.append(np.concatenate((rightConfig[0:6],leftConfig[0:6])))

SPos, SOrn = pybullet.getBasePositionAndOrientation(robotID)
print(")_(",SPos)

link_states=pybullet.getLinkStates(robotID,[5,11])
# print("***********",link_states[0][0],link_states[1][0],"******************")

rightF,leftF=np.zeros((int(240 * (rF.shape[0] - 2) * planner.tStep_),3)),np.zeros((int(240 * (rF.shape[0] - 2) * planner.tStep_),3))


#new for cheking whether small diversion matters 
for i in range(int(240 * (rF.shape[0] - 2) * planner.tStep_)):
    com_trajectory[i][2]=0.7

import pandas as pd
## convert your array into a dataframe
df = pd.DataFrame (com_trajectory)
filepath = 'com_trajectory.xlsx'
df = pd.DataFrame (com_trajectory)
df.to_excel(filepath, index=False)

df = pd.DataFrame (right)
filepath = 'right.xlsx'
df = pd.DataFrame (right)
df.to_excel(filepath, index=False)

df = pd.DataFrame (left)
filepath = 'left.xlsx'
df = pd.DataFrame (left)
df.to_excel(filepath, index=False)


N=int(240 * (rF.shape[0] - 2) * planner.tStep_)
def cal_derivative(param):
    print("______")
    print(param)
    der=np.zeros((N,3))
    for i in range(1,N):
        for j in range(3):
            der[i][j]=(param[i][j]-param[i-1][j])

    return der

print("*****")
print(com_trajectory)
plt.figure()
com_der=cal_derivative(np.array(com_trajectory))
plt.plot(com_der)
plt.legend(["x","y","z"])
plt.title("der_com")

plt.figure()
der=cal_derivative(np.array(right))
plt.plot(der)
plt.legend(["x","y","z"])
plt.title("right der")

plt.figure()
der=cal_derivative(np.array(left))
plt.plot(der)
plt.legend(["x","y","z"])
plt.title("left der")

plt.show()


for i in range(int(240 * (rF.shape[0] - 2) * planner.tStep_)):
    for index in range (6):
        #print("i:", com_trajectory[i],"\n",right[i],left[i])
        All = surena.doIK(com_trajectory[i], np.eye(3),left[i], np.eye(3),right[i], np.eye(3))
        leftConfig = All[6:12]
        rightConfig = All[0:6]
        pybullet.setJointMotorControl2(bodyIndex=robotID,
                                jointIndex=index,
                                controlMode=pybullet.POSITION_CONTROL,
                                targetPosition = rightConfig[index])
        pybullet.setJointMotorControl2(bodyIndex=robotID,
                                jointIndex=index + 6,
                                controlMode=pybullet.POSITION_CONTROL,
                                targetPosition = leftConfig[index])
        pybullet.stepSimulation()

        link_states=pybullet.getLinkStates(robotID,[5,11])
        z_l5,z_l11=link_states[0][0],link_states[1][0]
        rightF[i]=np.array(z_l5)-np.array([0.0054,0.0012,0])
        leftF[i]=np.array(z_l11)-np.array([0.0054,0.0012,0])
       

        ####
        #print("****", pybullet.getBasePositionAndOrientation(1))

        observation, iscontact, powers, x= Observations(robotID,planeID) 


        if not iscontact:
            up+=1
        else:
            up=0

        done=(observation[2*num_actions+1]<0.5) or (up>=2) #or .... ????
        step_counter+=1
        # if not iscontact:
        #     print("SC:",step_counter)

        # if done:     
        #     print("$$$$$$$$$DONE$$$$$$$$$$$$$$",observation[2*num_actions+1],up )
        
 
        temp=[powers,x,observation[2*num_actions],observation[2*num_actions+1],observation[2*num_actions+2],
        observation[2*num_actions+3],observation[2*num_actions+4],observation[2*num_actions+5],
        observation[2*num_actions+6]] #p,x,y,z,x.,y.,z.,F,F

        Dtheta.append(observation[num_actions:2*num_actions])
     
        #IMPORTANT: w2*done is not acurate, it should be fall instead but for now their are the same
        #reward= w0*(self.step_counter-50) + w1*(x-X0) - w2*(done) -w3*(powers) -w4*(observation[2*self.num_actions+1]-Z0) -w5*(observation[2*self.num_actions]) # x is not in observation
        #reward= w0*(step_counter-50) + w1*(x-X0) - w2*(done) -w5*(observation[2*num_actions]) # x is not in observation
        reward= w0*(step_counter-50) + w1*(x-X0) - w2*(done) -w3*(powers) -w4*(observation[2*num_actions+1]-Z0) -w5*(observation[2*num_actions]) # x is not in observation
        #print("reward:",reward)
        OBS.append(temp)
        

import pandas as pd

df = pd.DataFrame(OBS)
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='welcome', index=False)
writer.save()

pos=[]
vel=[]
f=[]
power=[]
for tmp in OBS:
    power.append(tmp[0])
    pos.append(tmp[1:4])
    vel.append(tmp[4:7])
    f.append(tmp[7:9])
    
    

import matplotlib.pyplot as plt
plt.figure()
plt.plot(power)
plt.figure()
plt.plot(pos)
plt.legend(["x","y","z"])
plt.figure()
plt.plot(vel)
plt.legend(["xd","yd","zd"])
plt.figure()
plt.plot(Dtheta)
plt.legend(["dx","yd","zd"])
#plt.plot(OBS,"*")

#plt.figure()
#plt.legend(["power","x","y","z","xd","yd","zd","F1","F2"])
plt.show()


# print("KNEE",KNEE[-1])

# textfile = open("file.txt", "w")
# for element in KNEE:
#     textfile.write(element + "\n")
# textfile.close()


rightF=rightF.T
leftF=leftF.T
com_trajectory=com_trajectory.T

plt.figure()
plt.plot(rightF[0],rightF[1])
plt.plot(leftF[0],leftF[1])
plt.plot(com_trajectory[0],com_trajectory[1])
plt.legend(["r","l","com"])

plt.show()





