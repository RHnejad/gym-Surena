import gym
from gym import spaces

import pybullet as p
import numpy as np

import time
import pybullet_data

ACTION=[]

TENDOF=1
DELTA_THETA =0
ACTIVATION=0

file_name="/content/gym-Surena/gym_Surena/envs/SURENA/sfixed.urdf"#google_colab_!git clone https://github.com/RHnejad/gym-Surena.git
#file_name="SURENA/surenaFIXEDtoWorld.urdf"
#file_name="SURENA/sfixed.urdf" if TENDOF else "SURENA/sfixed12.urdf" 

w0=10
w1,w2=20,20
w3,w4,w5=0.0001,2,2
    
X0=-0.517
Z0=0.9727
foot_z0=0.03799
T=200.


class SurenaRobot(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SurenaRobot, self).__init__()

    self.currentPos=np.zeros(((10 if TENDOF else 12),))

    if TENDOF:
      self.joint_space = {
            "low":np.array([ -0.5,-1.0,0.0,-1.0,-0.7,  -0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32),
            "high":np.array([0.1,1.2,2.0,1.3,0.7,   0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32)} 

      self.observation_space = gym.spaces.box.Box(
            low=np.array([-0.5,-1.0,0.0,-1.0,-0.7,  -0.1,-1.0,0.0,-1.0,-0.7,
            -315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,
            -2,-10,-1.5,-3.0,-3.0,-3.0,-600,-600], dtype=np.float32),
            high=np.array([0.1,1.2,2.0,1.3,0.7,   0.5,1.2,2.0 ,1.3,0.4,
            315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,
            100,10,1.5,3.0,3.0,3.0,600,600], dtype=np.float32))

             
      self.action_space = gym.spaces.box.Box(
            low=np.array([-0.0131,-0.0199,-0.0314,-0.0262,-0.0262   ,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262 ], dtype=np.float32),
            high=np.array([0.0131,0.0199,0.0314,0.0262,0.0262    ,0.0131,0.0199,0.0314,0.0262,0.0262   ], dtype=np.float32)) if DELTA_THETA else gym.spaces.box.Box(

            low=np.array([ -0.5,-1.0,0.0,-1.0,-0.7,  -0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32),
            high=np.array([0.1,1.2,2.0,1.3,0.7,   0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32))
      
    else:
      self.joint_space = {      
            "low":np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32),
            "high":np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32)}

      self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7,
            -315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,
            -2,-10,-1.5,-3.0,-3.0,-3.0,-600,-600], dtype=np.float32),
            high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4,
            315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,
            100,10,1.5,3.0,3.0,3.0,600,600], dtype=np.float32))
             
      self.action_space = gym.spaces.box.Box(
            low=np.array([-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262 ,-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262], dtype=np.float32),
            high=np.array([0.0131,0.0131,0.0199,0.0314,0.0262,0.0262 ,0.0131,0.0131,0.0199,0.0314,0.0262,0.0262], dtype=np.float32)) if DELTA_THETA else gym.spaces.box.Box(

            low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32),
            high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32))
        

    self.observation_dimensions= 28 if TENDOF else 32 
    self.rightLegID=[1,2,3,4,5]  if TENDOF else[0,1,2,3,4,5] 
    self.leftLegID=[7,8,9,10,11]  if TENDOF else[6,7,8,9,10,11]
    self.jointIDs=[1,2,3,4,5,7,8,9,10,11]  if TENDOF else[0,1,2,3,4,5,6,7,8,9,10,11] 
    self.num_actions= (10 if TENDOF else 12) 

    self.physicsClient = p.connect(p.DIRECT) #p.DIRECT for non-graphical version /// p.GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

    self.step_counter=0
    self.first_step=True
    self.up=0
    # g=-9.81
    # p.setGravity(0,0,g)
    self.reset()


  def Observations(self,SurenaID,planeId): 

    Ts=np.zeros(self.num_actions)
    Theta_dots=np.zeros(self.num_actions)
    Powers=np.zeros(self.num_actions)

    SPos, SOrn = p.getBasePositionAndOrientation(SurenaID)
    LinearVel,AngularVel=p.getBaseVelocity(SurenaID)
    JointStates=p.getJointStates(SurenaID,self.jointIDs) ##JPos,JVel,JF
    contacts=p.getContactPoints(SurenaID,planeId)
    iscontact=bool(len(contacts))

    link_states=p.getLinkStates(SurenaID,[5,11])
    z_l5,z_l11=link_states[0][0][2],link_states[1][0][2]
    if z_l5>=0.04 and z_l11>=0.04 :
        iscontact=False
    elif z_l5<=foot_z0 or z_l11<=foot_z0:
        iscontact=True

    observation_new=np.zeros(self.observation_dimensions)
    for ii in range(self.num_actions):
        observation_new[ii]=JointStates[ii][0] #theta
        observation_new[ii+self.num_actions]=JointStates[ii][1] #theta_dot
        
    FzR,FzL=0.0,0.0
    ncontact=len(contacts)
    for k in range(ncontact):
        if contacts[k][3]==11:
            FzR+=contacts[k][9]
        elif contacts[k][3]==5:
            FzL+=contacts[k][9]


    #FIX FOR 12 DOF

    x=SPos[0]

    #without x 
    # self.currentPos=observation_new[0:self.num_actions]    
    # observation_new[2*self.num_actions:2*self.num_actions+2]=np.array(SPos)[1:3]
    # observation_new[2*self.num_actions+2:2*self.num_actions+5]=np.array(LinearVel)
    # observation_new[2*self.num_actions+5:2*self.num_actions+7]=np.array([FzR, FzL]) #F_z_r and F_z_l
    # #observation_new[26:28]=np.array([JointStates[3][2][2] , JointStates[8][2][2]]) #F_z_r and F_z_l

    #with x
    self.currentPos=observation_new[0:self.num_actions]    
    observation_new[2*self.num_actions:2*self.num_actions+3]=np.array(SPos)
    observation_new[2*self.num_actions+3:2*self.num_actions+6]=np.array(LinearVel)
    observation_new[2*self.num_actions+6:2*self.num_actions+8]=np.array([FzR, FzL]) #F_z_r and F_z_l


    for jj in range(self.num_actions):
        Ts[jj]=JointStates[jj][3]
        Theta_dots[jj]=JointStates[jj][1]
        
    powers=sum(Ts*Theta_dots)
    powers=np.absolute(powers)
        
    return observation_new, iscontact, powers, x


  def step(self, action):
    
    time.sleep(1./T) #default=1./240. ################################################################################################################

    if ACTIVATION:
        action=(np.devide((self.action_space.high-self.action_space.low),2))*action
    action=action+self.currentPos if DELTA_THETA else action

    for i in range(self.num_actions):
      if action[i]<self.joint_space["low"][i]:
        action[i]=self.joint_space["low"][i]
      elif action[i]>self.joint_space["high"][i]:
        action[i]=self.joint_space["high"][i]

    p.setJointMotorControlArray(bodyUniqueId=1,
                                jointIndices=self.jointIDs,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions = action) #targetPositions =action[0].numpy())

    p.stepSimulation() #NOT SURE WHERE TO PLACE IT

    observation, iscontact, powers, x=self.Observations(1,0)   

    if not iscontact:
      self.up+=1
    else:
      self.up=0

    done=(observation[2*self.num_actions+1]<0.5) or (self.up>=5) #or .... ????
    if not done:
      self.step_counter+=1
     
    #IMPORTANT: w2*done is not acurate, it should be fall instead but for now their are the same
    #reward= w0*(self.step_counter-50) + w1*(x-X0) - w2*(done) -w3*(powers) -w4*(observation[2*self.num_actions+1]-Z0) -w5*(observation[2*self.num_actions]) # x is not in observation
    
    #without x
    #reward= w0*(self.step_counter-50) + w1*(x-X0) - w2*(done) -w5*(observation[2*self.num_actions]) # x is not in observation

    #with x
    reward= w0*(self.step_counter-50) + w1*(x-X0) - w2*(done) -w5*(observation[2*self.num_actions+1]) 

    if self.first_step:
      ACTION.append("*")
    ACTION.append(action)
    self.first_step=False

    return observation, reward, done, {}


  def reset(self):
    self.first_step=True
    print("step_counter: ",self.step_counter)
    self.step_counter=0
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    p.resetSimulation()
    obs=np.zeros(self.observation_dimensions)
    obs[2*self.num_actions:2*self.num_actions+2]=[0.0,Z0] #ignored x if not: :2*self.num_actions+2 and [X0,0.0,Z0]
    obs[self.observation_dimensions-2:self.observation_dimensions]=[-200,-200] #???????????
    planeId = p.loadURDF("plane.urdf") #ID:0
    Sid=p.loadURDF(file_name,startPos, startOrientation) #ID:1
    # p.enableJointForceTorqueSensor(Sid,4)
    # p.enableJointForceTorqueSensor(Sid,10)
    p.setGravity(0,0,-9.81)

    ###################################################################################################################################################
    p.setTimeStep(1./T)
    p.stepSimulation()
    
    return obs


  def render(self, mode='human'):
    pass
    #print("TEST")


  def close (self):
    p.disconnect()
    print(p.getConnectionInfo())

