import gym
from gym import spaces
import pybullet as p
import numpy as np
import time
import pybullet_data

ACTION=[]
REWS=np.array([])
save_actions=0

TENDOF=0
DELTA_THETA =1
ACTIVATION=0
KNEE=1
NORMALIZE=1
WITH_GUI =1
SAVE_MODEL=1


#file_name="/content/gym-Surena/gym_Surena/envs/SURENA/sfixed.urdf"#google_colab_!git clone https://github.com/RHnejad/gym-Surena.git
file_name="SURENA/sfixed.urdf"
#file_name="SURENA/sfixed.urdf" if TENDOF else "SURENA/sfixed12.urdf" 

    
X0=-0.517
Z0=0.9727
foot_z0=0.03799
T=50.

if KNEE:
    from SURENA.kasra.Robot import *
    from SURENA.kasra.DCM import *
    surena = Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)
    planner = DCMPlanner(0.7, 1.5, 0.45,1./200.) 


class SurenaRobot(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SurenaRobot, self).__init__()

        VelList=[25,25,38,60,50,50,25,25,38,60,50,50]

        self.theta_high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32)
        self.theta_low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32)

        self.thetaDot_high=high=np.multiply (np.array([0.0131,0.0131,0.0199,0.0314,0.0262,0.0262 ,0.0131,0.0131,0.0199,0.0314,0.0262,0.0262], dtype=np.float32),200./T)
        self.thetaDot_low=np.multiply (np.array([-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262 ,-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262], dtype=np.float32),200./T)            
        
        self.obs_high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4,
                315,315,315,315,315,315, 315,315,315,315,315,315,
                60,60,40,72,27,27, 60,60,40,72,27,27,
                10,1.5,3.0,3.0,3.0,
                1.0,1.0,1.0,1.0, 5.0,5.0,5.0,
                600,600], dtype=np.float32)
        self.obs_low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7,
                -315,-315,-315,-315,-315,-315, -315,-315,-315,-315,-315,-315,
                -60,-60,-40,-72,-27,-27,-60,-60,-40,-72,-27,-27,
                -10,-1.5,-3.0,-3.0,-3.0,
                -1.0,-1.0,-1.0,-1.0, -5.0,-5.0,-5.0,
                -600,-600], dtype=np.float32)

        
        if TENDOF:
            obs_high,obs_low=np.delete(self.obs_high,[0,6,12,18,24,30]),np.delete(self.obs_low,[0,6,12,18,24,30])
            theta_high,theta_low=np.delete(self.theta_high,[0,6]),np.delete(self.theta_low,[0,6])
            thetaDot_high,thetaDot_low=np.delete(self.thetaDot_high,[0,6]),np.delete(self.thetaDot_low,[0,6])
            VelList=np.delete(VelList,[0,6])

        #________________________________________
        
        self.observation_space = gym.spaces.box.Box(
                low=self.obs_low,
                high=self.obs_high)

        self.action_space=gym.spaces.box.Box(
                low=self.thetaDot_low,
                high=self.thetaDot_high) if DELTA_THETA else gym.spaces.box.Box(
                low=self.theta_low,
                high=self.theta_high)

        self.joint_space = {
                "low":self.theta_low,
                "high":self.theta_high} 

         
        self.maxVel=np.multiply(np.array(VelList, dtype=np.float32),2*(np.pi)/60)
        self.minVel=np.multiply(self.maxVel,-1)

        self.num_actions= (10 if TENDOF else 12)
        self.observation_dimensions= 3*self.num_actions+14 #28 if TENDOF else 32

        self.currentPos=np.zeros((self.num_actions,))
            
        self.rightLegID=[1,2,3,4,5]  if TENDOF else list(range(6))
        self.leftLegID=[7,8,9,10,11]  if TENDOF else list(range(6,12))
        self.jointIDs=self.rightLegID+self.leftLegID
            
        self.physicsClient = p.connect(p.GUI) if WITH_GUI else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        self.step_counter=0
        self.first_step=True
        self.up=0
        self.planeId=0
        self.SurenaID=1
        # g=-9.81
        # p.setGravity(0,0,g)
        self.reset()

    #________________________________________

    def Observations(self): 
        Ts_raw=np.zeros(self.num_actions)
        Theta_dots=np.zeros(self.num_actions)
        Powers=np.zeros(self.num_actions)

        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        LinearVel,AngularVel=p.getBaseVelocity(self.SurenaID)
        JointStates=p.getJointStates(self.SurenaID,self.jointIDs) ##JPos,JVel,JF
        contacts=p.getContactPoints(self.SurenaID,self.planeId)
        iscontact=bool(len(contacts))

        link_states=p.getLinkStates(self.SurenaID,[5,11])
        z_l5,z_l11=link_states[0][0][2],link_states[1][0][2]

        if z_l5>=0.04 and z_l11>=0.04 :
            iscontact=False
        elif z_l5<=foot_z0 or z_l11<=foot_z0: #double checks and if iscontact was decided as False but in fact ther is contact, corrects it
            iscontact=True

        FzR,FzL=0.0,0.0
        ncontact=len(contacts)
        for k in range(ncontact):
            if contacts[k][3]==11:
                FzR+=contacts[k][9]
            elif contacts[k][3]==5:
                FzL+=contacts[k][9]

        for jj in range(self.num_actions):
            Ts_raw[jj]=JointStates[jj][3]
            Theta_dots[jj]=JointStates[jj][1]
            
        Ts=np.absolute(Ts_raw)
        Theta_dots=np.absolute(Theta_dots)
        powers=sum(Ts*Theta_dots)

        x=SPos[0]

        observation_new=np.zeros(self.observation_dimensions)

        for ii in range(self.num_actions):
            observation_new[ii]=JointStates[ii][0] #theta
            observation_new[ii+self.num_actions]=JointStates[ii][1] #theta_dot
            
        #without x 
        self.currentPos=observation_new[0:self.num_actions] 

        observation_new[2*self.num_actions: 3*self.num_actions]=Ts_raw
        observation_new[3*self.num_actions: 3*self.num_actions+2]=np.array(SPos)[1:3]
        observation_new[3*self.num_actions+2: 3*self.num_actions+5]=np.array(LinearVel)

        observation_new[3*self.num_actions+5: 3*self.num_actions+9]=np.array(SOrn)
        observation_new[3*self.num_actions+9: 3*self.num_actions+12]=np.array(AngularVel)

        observation_new[3*self.num_actions+12: 3*self.num_actions+14]=np.array([FzR, FzL]) #F_z_r and F_z_l


        return observation_new, iscontact, powers, x

    #________________________________________

    def step(self, action):

        time.sleep(1./T)

        if ACTIVATION:
            action=(np.divide((self.action_space.high-self.action_space.low),2))*action
        action=action+self.currentPos if DELTA_THETA else action

        for i in range(self.num_actions):
            if action[i]<self.joint_space["low"][i]:
                action[i]=self.joint_space["low"][i]
            elif action[i]>self.joint_space["high"][i]:
                action[i]=self.joint_space["high"][i]

        p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                    jointIndices=self.jointIDs,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions = action
                                    )  #forces=np.full((self.num_actions,),600)
        p.stepSimulation() 

        observation, iscontact, powers, x=self.Observations() 
        

        if not iscontact:
            self.up+=1
        else:
            self.up=0

        done=(observation[3*self.num_actions+1]<0.4) #or (self.up>=20) #or .... ????

        if not done:
            self.step_counter+=1
            
        #[0.x,1.x_dot,2.stepCount,  3.done,4.power,5.dy,6.dz]
        param=np.array([observation[3*self.num_actions+2],
            powers, 
            (np.abs(observation[3*self.num_actions])**2), 
            (np.abs(observation[3*self.num_actions+1]-Z0))**2,
            (self.step_counter/num_steps)])

        #weights=np.array([+ 0.1 ,+5.0 , +500.0 ,  -6.0 , -1.0 ,-1.0 ,-1.0])
        weights=np.array([ +5.0 , -0.001 ,-7.0 ,-7.0, 10.])

        #heree
        reward_array=param*weights
        reward_s=(sum(reward_array)+1.625*(float(not done)))
        #print("reward:",reward_array)

        if done:
            print(self.step_counter)

        if save_actions:
            if self.first_step:
                ACTION.append("*")
            ACTION.append(action.tolist())

        self.first_step=False

        if NORMALIZE:
            temp=self.obs_high-self.obs_low
            observation=observation/temp
            
        return observation, reward_s, done, {}

    #________________________________________

    def knee(self):
        tha=[-0.,-0.28140172, -0.93511444,  0., 0.93511444,  0.28140172,
        -0., -0.55515457, -0.87560747,  0., 0.87560747,  0.55515457]
        p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                    jointIndices=[0,1,2,3,4,5,6,7,8,9,10,11] ,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions = tha) 


    def bend_knee(self):
        for i in range(240):
                All = surena.doIK([0.0,0.0,0.74 - (i/240)* (0.74-planner.deltaZ_)], np.eye(3),[0.0,0.115,0.0], np.eye(3),[0.0, -0.115,0.0], np.eye(3))
                leftConfig = All[6:12]
                rightConfig = All[0:6]           
                for index in range (6):
                    p.setJointMotorControl2(bodyIndex=self.SurenaID,
                                            jointIndex=index,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition = rightConfig[index])
                    p.setJointMotorControl2(bodyIndex=self.SurenaID,
                                            jointIndex=index + 6,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition = leftConfig[index])
                p.stepSimulation()
        time.sleep(0.25)

    #________________________________________

    def reset(self):
        self.first_step=True
        self.step_counter=0
        startPos = [0,0,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        p.resetSimulation()
        self.planeId = p.loadURDF("plane.urdf") #ID:0
        self.SurenaID=p.loadURDF(file_name,startPos, startOrientation) 
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1./T)
        if KNEE:
            self.bend_knee()

        # obs=np.zeros(self.observation_dimensions)
        # obs[2*self.num_actions:2*self.num_actions+2]=[0.0,Z0] #ignored x if not: :2*self.num_actions+2 and [X0,0.0,Z0]
        # obs[self.observation_dimensions-2:self.observation_dimensions]=[-200,-200] #???????????
        # p.enableJointForceTorqueSensor(self.SurenaID,4)
        # p.enableJointForceTorqueSensor(self.SurenaID,10)
        
        p.stepSimulation()
        obs=self.Observations()[0]
        if NORMALIZE:
            temp=self.obs_high-self.obs_low
            obs=obs/temp
        
        return obs

    #________________________________________
    def render(self, mode='human'):
        pass
    def close (self):
        p.disconnect()
        print(p.getConnectionInfo())
    #________________________________________
