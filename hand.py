
#PARAMETERS TO CHANGE: ____________________________
N=800
urfdName="SURENA/surenaNOTWorld.urdf"  #surenaFIXEDtoWorld.urdf
txtName="qscenario.txt"
RightArm=True
Plot=True
#__________________________________________________


from os import MFD_HUGE_64KB
import numpy as np
import pybullet as p
import time
import pybullet_data

#Reading inputs from the file
with open("qscenario.txt") as f:
    lines = f.readlines()
    
f.close()

M=np.zeros(N*7)
i=0
for number in lines:
    M[i]=float(number)
    i+=1

#_____________

def Reset():
    
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    p.resetSimulation()
    planeId = p.loadURDF("plane.urdf") 
    Sid=p.loadURDF(urfdName, useFixedBase=True)
    p.setGravity(0,0,-9.81)
    p.setTimeStep(1./200.)
    p.stepSimulation()

    return Sid,planeId

#_____________

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

joints=list(range(14,21)) if RightArm else list(range(21,28)) 

feedback_theta=np.zeros([N,7])
feedback_thetaDot=np.zeros([N,7])
actions=np.zeros([N,7])

#_____________
i=0
j=0
flag=True
Sid,planeId=Reset()
while flag:

    time.sleep(1./200.)

    p.setJointMotorControlArray(bodyUniqueId=Sid,
                                jointIndices=joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions = M[j:j+7]
                                ) #forces=np.full((7,),10.)
    p.stepSimulation()
    
    actions[i]=M[j:j+7]
    JointStates=p.getJointStates(Sid,joints)
    for ii in range(7):
        feedback_theta[i][ii]=JointStates[ii][0] #theta
        feedback_thetaDot[i][ii]=JointStates[ii][1] #theta_dot
        

    j+=7
    i+=1
    if i==N:    
        print("_____ Press Y if you want to replay _____")
        inp=input()
        if (inp=="Y" or inp=="y"):
            Reset()
            i=0
            j=0
        else:
            flag=False
        


p.disconnect()





def plot_actions(): 
    plt.figure()
    plt.plot(actions)
    plt.legend([1,2,3,4,5,6,7])
    plt.title("Actions")
    

def plot_feedbacks():
    plt.figure()
    plt.plot(feedback_theta)
    plt.legend([1,2,3,4,5,6,7])
    plt.title("FeedBack Theta")

    plt.figure()
    plt.plot(feedback_thetaDot)
    plt.legend([1,2,3,4,5,6,7])
    plt.title("FeedBack Theta_Dot")

if Plot:
    import matplotlib.pyplot as plt
    plot_actions()
    plot_feedbacks()
    plt.show()

