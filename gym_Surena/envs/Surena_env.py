import gym
from gym import spaces

import pybullet as p
import numpy as np

import time
import pybullet_data

class SurenaRobot(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SurenaRobot, self).__init__()
    # # Define action and observation space
    # # They must be gym.spaces objects
    # # Example when using discrete actions:
    # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # # Example for using image as input:
    # self.observation_space = spaces.Box(low=0, high=255,
    #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)


    self.action_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0], dtype=np.float32),
            high=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 ], dtype=np.float32))
    self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
            -315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,-315.0,
            -2.0,-1000,-1.5,-3.0,-3.0,-3.0,-600,-600], dtype=np.float32),
            high=np.array([1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
            315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,315.0,
            1000,1000,1.5,3.0,3.0,3.0,600,600], dtype=np.float32))



    self.observation_dimensions=28
    self.rightLegID=[1,2,3,4,5]
    self.leftLegID=[7,8,9,10,11]
    self.jointIDs=[1,2,3,4,5,7,8,9,10,11] #new end
    self.num_actions=10


    self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

    g=-9.81
    p.setGravity(0,0,g)

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

    observation_new=np.zeros(self.observation_dimensions)
    for ii in range(10):
        observation_new[ii]=JointStates[ii][0]
        observation_new[ii+10]=JointStates[ii][1]
        
    FzR,FzL=0.0,0.0
    ncontact=len(contacts)
    for k in range(ncontact):
        if contacts[k][3]==11:
            FzR+=contacts[k][9]
            print("FR:"+str(contacts[k][9]))
        elif contacts[k][3]==5:
            FzL+=contacts[k][9]
            print("FZ:"+str(contacts[k][9]))

        # if ep<3:
        #     print("^^^^^")
        #     print(FzR)
        #     print(FzR)
        #     print(contacts)
        
        
    observation_new[20:23]=np.array(SPos)
    observation_new[23:26]=np.array(LinearVel)
    observation_new[26:28]=np.array([FzR, FzL]) #F_z_r and F_z_l
    #observation_new[26:28]=np.array([JointStates[3][2][2] , JointStates[8][2][2]]) #F_z_r and F_z_l


    for jj in range(10):
        Ts[jj]=JointStates[jj][3]
        Theta_dots[jj]=JointStates[jj][1]
        
    powers=sum(Ts*Theta_dots)
    powers=np.absolute(powers)
        

    return observation_new, iscontact, powers

      




  def step(self, action):

    w1,w2=20,20
    w3,w4,w5=0.0001,2,2
    
    x0=-0.517
    z0=0.9727

    time.sleep(1./240.) #default=1./240.

    p.setJointMotorControlArray(bodyUniqueId=1,
                                jointIndices=self.jointIDs,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions =action) #targetPositions =action[0].numpy())

    p.stepSimulation() #NOT SURE WHERE TO PLACE IT

    observation, iscontact, powers=self.Observations(1,0)   


    done=(observation[22]<0.5) or (not iscontact) #or .... ????
        
    #IMPORTANT: w2*done is not acurate, it should be fall instead but for now their are the same
    reward= w1*(observation[20]-x0) - w2*(done) -w3*(powers) -w4*(observation[22]-z0) -w5*(observation[21])


    #...
    return observation, reward, done, {}


  def reset(self):

    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    p.resetSimulation()
    obs=np.zeros(self.observation_dimensions)
    obs[20:23]=[-0.517,0.0,0.9727]
    obs[26:28]=[-200,-200] #???????????
    planeId = p.loadURDF("plane.urdf") #ID:0
    Sid=p.loadURDF("SURENA/sfixed.urdf",startPos, startOrientation) #ID:1
    p.enableJointForceTorqueSensor(Sid,4)
    p.enableJointForceTorqueSensor(Sid,10)
    p.setGravity(0,0,-9.81)
    p.stepSimulation()
    
    return obs
    #...
    #return observation  # reward, done, info can't be included



  def render(self, mode='human'):
    print("TEST")
    #...
  def close (self):
    p.disconnect()
    print(p.getConnectionInfo())
    #...
