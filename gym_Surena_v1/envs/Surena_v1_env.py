import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import time
from matplotlib.pyplot import flag
from SURENA.Robot import *

WITH_GUI=1
KNEE=1
ACTIVATE_SLEEP=0
if not WITH_GUI: ACTIVATE_SLEEP=False
SCALE_ACTIONS=1


#"SURENA/sfixedWLess.urdf"
#"SURENA/sfixedWless6dof.urdf"

#file_name="/content/gym-Surena/gym_Surena/envs/SURENA/sfixed.urdf"#google_colab_!git clone https://github.com/RHnejad/gym-Surena.git
file_name="SURENA/sfixed.urdf"
#file_name="SURENA/sfixed.urdf" if TENDOF else "SURENA/sfixed12.urdf" 

deltaS=0.5    
X0=-0.517
Z0=0.9727
foot_z0=0.037999 
foor_y0_r=0.11380
T=200.

if KNEE: from SURENA.DCM import *
    

# global x
# x=0
global com_for_plot
com_for_plot=[[],[]]


class SurenaRobot_v1(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SurenaRobot_v1, self).__init__()

        self.action_min=np.array([0,-0.0018,0,-0.0007,0],dtype=np.float32)
        self.action_max=np.array([0.0015,0.0014,0.004,0.0007,3],dtype=np.float32)

        self.observation_space = gym.spaces.box.Box(
                low=np.array([0,-0.0018,0,-0.0007,0,-0.5,-0.5],dtype=np.float32),
                high=np.array([0.0015,0.0014,0.004,0.0007,3,5,0.5],dtype=np.float32))

        #[com_x,com_y,ankle_x,ankle_y,(DS,R,L)]
        self.action_space=gym.spaces.box.Box(
                low= np.negative(np.ones(5,dtype=np.float32)) if SCALE_ACTIONS else self.action_min,
                high= np.ones(5,dtype=np.float32) if SCALE_ACTIONS else self.action_max ) 
                       
        self.num_actions= 5
        self.observation_dimensions= 7
            
        self.rightLegID=list(range(6))
        self.leftLegID=list(range(6,12))
        self.jointIDs=self.rightLegID+self.leftLegID
        self.startPos = [0,0,0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.last_actions=np.array([0.0,0.0,0.0,0.0,0.0])
        self.right,self.left=np.array([0,-0.115,0]),np.array([0,0.115,0])
            
        self.physicsClient = p.connect(p.GUI) if WITH_GUI else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.surena=Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)

        self.observations=np.zeros(self.observation_dimensions)

        if KNEE: self.planner = DCMPlanner(0.7, 1.5, 0.45,1./T) 
        
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

    #________________________________________
    
    def step(self, action):
        
        if SCALE_ACTIONS: action=(np.divide((self.action_max-self.action_min),2))*action+(np.divide((self.action_max+self.action_min),2))
        com=np.array([action[0]+self.last_actions[0],action[1]+self.last_actions[1],0.7])
        feet_state=action[-1]
        if feet_state<2 and feet_state>=1: 
            self.right=np.array([action[2]+self.last_actions[2],-0.115,action[3]+self.last_actions[3]])       
        elif feet_state<3 and feet_state>=2:
            self.left=np.array([action[2]+self.last_actions[2],0.115,action[3]+self.last_actions[3] ])    
        
        All = self.surena.doIK(com, np.eye(3),self.left, np.eye(3),self.right, np.eye(3))
        p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                        jointIndices=self.jointIDs,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions = All)  #forces=np.full((self.num_actions,),600)
        p.stepSimulation()

        
        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        LinearVel,AngularVel=p.getBaseVelocity(self.SurenaID)  

        done= SPos[2]<0.5

        self.observations[0:self.num_actions]=action
        self.observations[5:7]=SPos[1:3]

        param=np.array([LinearVel[0],
            self.cal_power(), 
            (np.abs(SPos[1])**2), 
            (np.abs(SPos[2]-Z0))**2,
            (self.step_counter/1000)])

        weights=np.array([ +8.0 , -0.00001 ,-7.0 ,-1.0, 0. ])
        #heree
        reward_array=param*weights
        reward_s=(sum(reward_array)+1.625*(float(not done)))#-1.7*float(bool(self.up))

        global com_for_plot
        com_for_plot[0].append(action[0])
        com_for_plot[1].append(action[-1])


        #if done:
        #    plt.figure()
        #    plt.plot(com_for_plot[0],com_for_plot[1],"*")
        #    plt.show()
        #    com_for_plot=[[],[]]


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

        self.last_actions=np.array([0.0,0.0,0.0,0.0,0.0])
        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)

        obs=np.append(self.last_actions,SPos[1:3])
   
        return obs
    #________________________________________
    def render(self, mode='human'):
        pass
    def close (self):
        p.disconnect()
        print(p.getConnectionInfo())
    #________________________________________


if __name__ == "__main__":

  S=SurenaRobot_v1()
  for i in range(100000):
  	S.step([0,0,0,0,0])
  
  S.close()
