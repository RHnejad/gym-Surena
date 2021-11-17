
file_name="SURENA/sfixedlim.urdf"

import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import time
from matplotlib.pyplot import flag
from Robot import *

WITH_GUI=0
KNEE=1
MIN=1
ACTIVATE_SLEEP=0
if not WITH_GUI: ACTIVATE_SLEEP=False
SCALE_ACTIONS=0


deltaS=0.5    
X0=-0.517
Z0=0.9727
Z0_2=0.7
foot_z0=0.037999 
foor_y0_r=0.11380
T=200.
beta=50*200./T

if KNEE: from DCM import *
# global x
# x=0
global com_for_plot
com_for_plot=[[],[]]


class SurenaRobot_v1(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,gui=""):
        super(SurenaRobot_v1, self).__init__()
        if gui=="gui" : 
            global WITH_GUI
            WITH_GUI=1

        self.num_actions= 4 if MIN else 5
        self.observation_dimensions= self.num_actions+5

        #DeltaX_com,DeltaY_com,DeltaX_anke,(DeltaY_ankle),DeltaZ_ankle
        self.action_min=np.array([0, -0.0018, 0, -0.0007],dtype=np.float32) if MIN else np.array([0, -0.0018, 0, -0.004,-0.0007],dtype=np.float32)
        self.action_max=np.array([0.0015, 0.0014, 0.004, 0.0007],dtype=np.float32) if MIN else np.array([0.0015, 0.0014, 0.004,  0.004,0.0007],dtype=np.float32)
        self.action_min=np.multiply(self.action_min,beta)
        self.action_max=np.multiply(self.action_max,beta)


        self.observation_space = gym.spaces.box.Box(
                low=np.concatenate((self.action_min,[-0.1,-0.5,-0.5,  0,0])),
                high=np.concatenate((self.action_max,[5,0.5,0.9727, 250,250])))

        #[com_x,com_y,ankle_x,ankle_y,(DS,R,L)]
        self.action_space=gym.spaces.box.Box(
                low= np.negative(np.ones(self.num_actions,dtype=np.float32)) if SCALE_ACTIONS else self.action_min,
                high= np.ones(self.num_actions,dtype=np.float32) if SCALE_ACTIONS else self.action_max ) 
                                
        self.rightLegID=list(range(6))
        self.leftLegID=list(range(6,12))
        self.jointIDs=self.rightLegID+self.leftLegID
        self.startPos = [0,0,0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.last_actions=np.zeros(self.num_actions,dtype=np.float32)
        self.right,self.left=np.array([0,-0.115,0]),np.array([0,0.115,0])
        self.com,self.right_ankle,self.left_ankle=None,None,None
        self.up=0
            
        self.physicsClient = p.connect(p.GUI) if WITH_GUI else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.surena=Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)

        self.observations=np.zeros(self.observation_dimensions,dtype=np.float32)

        if KNEE: self.planner = DCMPlanner(Z0_2, 1.5, 0.45,1./T) 
        self.first_step=True
        self.reset()
#____________________________________________________________________________________________

    def cal_power(self):
        Ts_raw=np.zeros(self.num_actions)
        Theta_dots=np.zeros(self.num_actions)
        jointStates=p.getJointStates(self.SurenaID,self.jointIDs) 
        for jj in range(self.num_actions):
            Ts_raw[jj]=jointStates[jj][3]
            Theta_dots[jj]=jointStates[jj][1]
                
        Ts=np.absolute(Ts_raw)
        Theta_dots=np.absolute(Theta_dots)
        power=sum(Ts*Theta_dots)

        return power

    #___________________________

    def bend_knee(self):
        for i in range(240):
                All = self.surena.doIK([0.0,0.0,0.74 - (i/240)* (0.74-self.planner.deltaZ_)], np.eye(3),[0.0,0.115,0.0], np.eye(3),[0.0, -0.115,0.0], np.eye(3))         
                for index in range (6):
                    p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                        jointIndices=self.jointIDs,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions = All) 
                p.stepSimulation()
        if ACTIVATE_SLEEP:
            time.sleep(1)

#____________________

    def feet_forces(self):
        contacts=p.getContactPoints(self.SurenaID,self.planeId)
        iscontact=bool(len(contacts))
        if not iscontact:
            self.up+=1
        else:
            self.up=0

        FzR,FzL=0.0,0.0
        ncontact=len(contacts)
        for k in range(ncontact):
            if contacts[k][3]==5:
                FzR+=contacts[k][9]

            elif contacts[k][3]==11:
                FzL+=contacts[k][9]

        return FzR,FzL

    #________________
 
    def cal_trajectories(self,action):

        if SCALE_ACTIONS: action= (np.divide((self.action_max-self.action_min),2))*action +(np.divide((self.action_max+self.action_min),2))
        target_com=np.array([action[0]+self.com[0],action[1]+self.com[1],Z0_2])
        print("1",target_com)

        #feet_state=np.abs(self.right_ankle[1]-self.com[1]) > np.abs(self.left_ankle[1]-self.com[1]) #0:right 1:left on the floor
        feet_state=0
        #_______
        # if self.first_step:
        #     self.first_step=False
        #     while np.abs(self.observations[self.num_actions+1])<0.1166:
        #         temp=self.action_space.high[1] if feet_state else self.action_space.low[1]
        #         temp_ac=np.zeros(self.num_actions)
        #         temp_ac[1]=temp
        #         self.step(temp_ac)
        #______

        if feet_state: #left foot on the ground
            self.right=np.array([action[2]+self.right_ankle[0],self.right_ankle[1],action[3]+self.right_ankle[2]]) 
            self.left=np.array([self.left_ankle[0],self.left_ankle[1],0.0]) 
            if not MIN:  self.right=np.array([action[2]+self.right_ankle[0],action[3]+self.right_ankle[1],action[4]+self.right_ankle[2]]) 
            
        else:
            self.left=np.array([action[2]+self.left_ankle[0],self.left_ankle[1],action[3]+self.left_ankle[2]])  
            self.right=np.array([self.right_ankle[0],self.right_ankle[1],0.0])
            if not MIN:  self.left=np.array([action[2]+self.left_ankle[0],action[3]+self.left_ankle[1],action[4]+self.left_ankle[2]])

        return target_com 

    #________________________________________

    def step(self, action):

        com_trj=self.cal_trajectories(action)
        All = self.surena.doIK(com_trj, np.eye(3),self.left, np.eye(3),self.right, np.eye(3))
        p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                        jointIndices=self.jointIDs,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions = All)  #forces=np.full((self.num_actions,),600)
        p.stepSimulation()
        if ACTIVATE_SLEEP: time.sleep(1/T)
        # print(com_trj,self.right,self.left)

        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        LinearVel,AngularVel=p.getBaseVelocity(self.SurenaID)  
        ankle_states=p.getLinkStates(self.SurenaID,[4,10])
        self.right_ankle,self.left_ankle=np.array(ankle_states[0][0]),np.array(ankle_states[0][1])
        self.com=SPos[0:2]

        self.observations[0:self.num_actions]=action
        self.observations[self.num_actions:self.num_actions+3]=SPos
        self.observations[-2:]=self.feet_forces()

        done=float(bool(SPos[2]<0.5)) #or ( self.observations[-1:]>0 and self.left_ankle[0]>=0.2) #SPos[2]<0.5 or SPos[0]>2. or self.up>1 or 
        

        param=np.array([LinearVel[0],
            self.cal_power(), 
            max(0, np.exp(np.abs(SPos[1]-0.115))-1 ), #exp(delta_y-acceptable_delta_y)
            max(0,np.exp(np.abs(SPos[2]-Z0)-0.03)-1), 
            (self.step_counter/1000),
            SPos[0],
            self.left_ankle[0]])

        weights=np.array([ +0.0 , -0.00000 ,-0.1 ,-0.0, 0. , 0.3 ,0.7])
        #heree
        reward_array=param*weights
        # print("R",reward_array)
        # print("com",com)
        # print("All",All)
        reward_s=(sum(reward_array))-0.015-0.1*float(bool(self.up))-150*float(bool(SPos[2]<0.5))

        global com_for_plot
        com_for_plot[0].append(action[0])
        com_for_plot[1].append(action[-1])


        # if done:
        #    plt.figure()
        #    plt.plot(com_for_plot[0],com_for_plot[1],"*")
        #    plt.show()
        #    com_for_plot=[[],[]]
        self.first_step=False

        return self.observations, reward_s, done, {}

    #________________________________________

    def reset(self):
        self.first_step=True
        self.step_counter=0
        p.resetSimulation()
        self.planeId = p.loadURDF("plane.urdf") #ID:0
        self.SurenaID=p.loadURDF(file_name,self.startPos, self.startOrientation) 
        p.enableJointForceTorqueSensor(self.SurenaID,5)
        p.enableJointForceTorqueSensor(self.SurenaID,11)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1./T)

        if KNEE: self.bend_knee() 
        else: p.stepSimulation()
        
        self.last_actions=np.zeros(self.num_actions,dtype=np.float32)

        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        ankle_states=p.getLinkStates(self.SurenaID,[4,10])
        self.right_ankle,self.left_ankle=np.array(ankle_states[0][0]),np.array(ankle_states[1][0])
        self.com=SPos[0:2]

        self.observations[0:self.num_actions]=self.last_actions
        self.observations[self.num_actions:self.num_actions+3]=SPos
        self.observations[-2:]=self.feet_forces()
   
        return self.observations
    #________________________________________
    def render(self, mode='human'):
        pass
    def close (self):
        p.disconnect()
        print(p.getConnectionInfo())
    #________________________________________


if __name__=="__main__":
    S=SurenaRobot_v1()
    print(S.com)
    # print("*************",p.getLinkStates(S.SurenaID,S.jointIDs))
    
    # for i in range(10000):
    #     a,b,done,c=S.step([.1,.1,.1,.1])
    #     if done:
    #         S.reset()


    import json

    with open('data.txt') as json_file:
        data = json.load(json_file)
        for pr in data['people']:
            com_der=np.reshape(np.array(pr['COM']),(-1,3))
            der1=np.reshape(np.array(pr['R']),(-1,3))
            der2=np.reshape(np.array(pr['L']),(-1,3))

    plt.figure()
    plt.plot(com_der)
    plt.legend(["x","y","z"])
    plt.title("der_com")

    plt.figure()
    plt.plot(der1)
    plt.legend(["x","y","z"])
    plt.title("right der")

    plt.figure()
    plt.plot(der2)
    plt.legend(["x","y","z"])
    plt.title("left der")

    plt.show()
    # print(der1)
    # print(der2)

    for ii in range(21): #2160
        
        feet_state=np.abs(S.right_ankle[1]-S.com[1]) > np.abs(S.left_ankle[1]-S.com[1]) #0:right 1:left on the floor
        # feet_state=(der2[ii][0]==0)
        # if ii<20:print(S.right_ankle[1],S.com[1],S.left_ankle[1],S.com[1],feet_state)
        if feet_state:
            ac=np.array([com_der[ii][0],com_der[ii][1],der1[ii][0],der1[ii][2]])
        else:
            ac=np.array([com_der[ii][0],com_der[ii][1],der2[ii][0],der2[ii][2]])
        if ii<20:print("obs",S.observations[4:])
        S.step(ac)

    print("__end__")
