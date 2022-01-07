
file_name="SURENA/sfixedWLessLim.urdf"
# file_name="SURENA/newFootSfixedlim.urdf"

import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import time
from matplotlib.pyplot import flag
from Robot import *

num_steps=1800

WITH_GUI=0
KNEE=1
MIN=1
ACTIVATE_SLEEP=0
if not WITH_GUI: ACTIVATE_SLEEP=True
SCALE_ACTIONS=1
ONLY_COM=0
IMITATE=1

com_sensor_bias=np.array([+0.0517,0.0,-0.94428+0.7],dtype=np.float32) #-0.9727
right_ankle_sensor_bias=np.array([+0.029,+0.1167-0.115,-0.167],dtype=np.float32)
left_ankle_sensor_bias=np.array([+0.029,-0.1167+0.115,-0.167],dtype=np.float32)

deltaS=0.5    
X0=-0.0517
Z0=0.9727
Z0_2=0.7
foot_z0=0.037999 
foor_y0_r=0.11380
T=200.
beta=5*200./T

PLOT_REWS=1
N_plot=10000

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

        if PLOT_REWS: self.mean_reward_array=np.zeros((N_plot,8))
        self.episode_num=0 #zero base

        self.num_actions= (4 if MIN else 5) if not ONLY_COM else 2
        self.observation_dimensions= self.num_actions+5+12

        #DeltaX_com,DeltaY_com,DeltaX_anke,(DeltaY_ankle),DeltaZ_ankle
        if ONLY_COM:
            self.action_min=np.array([0, -0.0018],dtype=np.float32)
            self.action_max=np.array([0.0015, 0.0014],dtype=np.float32) 
        else:
            self.action_min=np.array([0, -0.0018, 0, -0.0007],dtype=np.float32) if MIN else np.array([0, -0.0018, 0, -0.004,-0.0007],dtype=np.float32)
            self.action_max=np.array([0.0015, 0.0014, 0.004, 0.0007],dtype=np.float32) if MIN else np.array([0.0015, 0.0014, 0.004,  0.004,0.0007],dtype=np.float32)
        self.action_min=np.multiply(self.action_min,beta)
        self.action_max=np.multiply(self.action_max,beta)
        
        self.observation_space = gym.spaces.box.Box(
                low=np.concatenate((self.action_min,[-0.1,-0.5,-0.5,  0,0, 
                                                    -1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7])),
                high=np.concatenate((self.action_max,[5,0.5,0.9727, 250,250,
                                                    0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4 ])))

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
        self.FzR,self.FzL=250,250
        self.foot_step_count=0
        self.feet_state=False
        self.reset()

        self.zmp=[0,0,0]

        self.step_counter=0

        if IMITATE:
            import json
            with open('classic200.txt') as json_file:
                data = json.load(json_file)
                for pr in data['robot']:
                    self.des_com=np.reshape(np.array(pr['come']),(-1,3))
                    self.des_right=np.reshape(np.array(pr['right']),(-1,3))
                    self.des_left=np.reshape(np.array(pr['left']),(-1,3))
                    self.des_theta=np.reshape(np.array(pr['theta']),(-1,12))

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

        tempR,tempL=0.0,0.0
        ncontact=len(contacts)
        for k in range(ncontact):
            if contacts[k][3]==5:
                tempR+=contacts[k][9]

            elif contacts[k][3]==11:
                tempL+=contacts[k][9]

        # print(self.FzL,self.FzR)

        if self.FzR-tempR<0 and self.feet_state==False:#right
            self.foot_step_count+=1
            # print("R",self.foot_step_count)
        elif self.FzL-tempL<0 and self.feet_state==True: #left
            self.foot_step_count+=1
            # print("L",self.foot_step_count)
        else:
            self.foot_step_count=0

        if self.foot_step_count>1:
            # print("*",self.feet_state)
            self.feet_state=(not self.feet_state) #switch
            # print("^",self.feet_state)
        
        self.FzR,self.FzL=tempR,tempL
        self.zmp=self.cal_ZMP()
        ZMP_in_SP=self.process_ZMP(contacts)

    #________________
    def cal_trajectories(self,action):

        if SCALE_ACTIONS: action= (np.divide((self.action_max-self.action_min),2))*action +(np.divide((self.action_max+self.action_min),2))
        
        target_com=np.array([action[0]+self.com[0],action[1]+self.com[1],Z0_2])
        target_com[:2]+=com_sensor_bias[:2]
       
        if self.feet_state and (not ONLY_COM): #left foot on the ground
            self.right=np.array([action[2]+self.right_ankle[0],self.right_ankle[1],action[3]+self.right_ankle[2]]) 
            self.left=np.array([self.left_ankle[0],self.left_ankle[1],0.0]) 
            if not MIN:  self.right=np.array([action[2]+self.right_ankle[0],action[3]+self.right_ankle[1],action[4]+self.right_ankle[2]]) 
        elif ONLY_COM:
            pass   
        else:
            self.left=np.array([action[2]+self.left_ankle[0],self.left_ankle[1],action[3]+self.left_ankle[2]])  
            self.right=np.array([self.right_ankle[0],self.right_ankle[1],0.0])
            if not MIN:  self.left=np.array([action[2]+self.left_ankle[0],action[3]+self.left_ankle[1],action[4]+self.left_ankle[2]])

        self.right+=right_ankle_sensor_bias
        self.left+=left_ankle_sensor_bias

        return target_com 


    def cal_observations(self,action):
        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        LinearVel,AngularVel=p.getBaseVelocity(self.SurenaID)  
        ankle_states=p.getLinkStates(self.SurenaID,[4,10])

        self.right_ankle,self.left_ankle=np.array(ankle_states[0][0]),np.array(ankle_states[1][0])
        self.com=SPos[0:2]
        
        #***
        # print(self.observation_dimensions,"&&&&")
        JointStates=p.getJointStates(self.SurenaID,self.jointIDs) ##JPos,JVel,JF
        for ii in range(12):
            # print(ii)
            self.observations[ii+self.num_actions+5]=JointStates[ii][0] #theta

        self.observations[0:self.num_actions]=action
        self.observations[self.num_actions:self.num_actions+3]=SPos

        self.feet_forces()
        self.observations[self.observation_dimensions-14:self.observation_dimensions-12]=[self.FzR,self.FzL]

        return SPos,LinearVel

    #_______________________________________________-
    def cal_rewards(self,com_trj, SPos, LinearVel):

        imitation_reward_theta=0
        # if IMITATE:
        #     imitation_reward_theta=np.power((self.observation[0:self.num_actions]-self.des_theta[self.step_counter%1800]),2)
        #     imitation_reward_theta=np.sum(imitation_reward_theta)
        #     imitation_reward_theta=np.exp(-1*imitation_reward_theta) #chenge -1 another negative num. if necessary

        imitation_reward=0
        if IMITATE:
            imitation_reward=np.power((com_trj[1]-self.des_com[self.step_counter%1800][1]),2)
            imitation_reward+=np.power((self.right[2]-self.des_right[self.step_counter%1800][2]),2)
            imitation_reward+=np.power((self.left[2]-self.des_left[self.step_counter%1800][2]),2)
            # imitation_reward=np.sum(imitation_reward)
            imitation_reward=np.exp(-2*(imitation_reward))
            # print(imitation_reward)

        param=np.array([LinearVel[0], #x_dot
        self.cal_power(), 
        max(0, np.exp(np.abs(SPos[1]-0.115))-1 ), #exp(delta_y-acceptable_delta_y)
        max(0,np.exp(np.abs(SPos[2]-Z0)-0.03)-1), 
        (self.step_counter/1000),
        SPos[0], #x
        self.left_ankle[0],
        imitation_reward])

        weights=np.array([ +.09 , -0.00000 ,-0.0 ,-0.0, 0. , 2.5 ,0.0 ,2.])  
        #heree
        reward_array=param*weights
        # print(reward_array)
        reward_s=(sum(reward_array))+1.25+self.foot_step_count*0.8#-0.1*float(bool(self.up))-150*float(bool(SPos[2]<0.5))
        reward_s=reward_s/2

        if PLOT_REWS :self.mean_reward_array[self.episode_num%N_plot]+=param
        
        return reward_s
    #________________________________________
    def step(self, action):
        com_trj=self.cal_trajectories(action)
        # print(self.left,self.right)
        All = self.surena.doIK(com_trj, np.eye(3),self.left, np.eye(3),self.right, np.eye(3))
        p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                        jointIndices=self.jointIDs,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions = All)  #forces=np.full((self.num_actions,),600)
        p.stepSimulation()
        if ACTIVATE_SLEEP: time.sleep(1/T)
        # print(com_trj)


        SPos,LinearVel=self.cal_observations(action)

        done=((SPos[2]<0.5))  or self.up>50 #or ( self.observations[-1:]>0 and self.left_ankle[0]>=0.2) #SPos[2]<0.5 or SPos[0]>2. or self.up>1 or 
        # print(SPos[2])
        reward=self.cal_rewards(com_trj, SPos, LinearVel)

        if not done: self.step_counter+=1   
        elif PLOT_REWS : 
            self.mean_reward_array[self.episode_num%N_plot]= np.divide(self.mean_reward_array[self.episode_num%N_plot],self.step_counter+1)
            self.episode_num+=1  
        if done and self.episode_num%N_plot==0 and self.episode_num>0 and PLOT_REWS:
            print(self.episode_num)
            fig=plt.figure()
            plt.plot(self.mean_reward_array)
            plt.show()
            plt.close()

        # global com_for_plot
        # com_for_plot[0].append(action[0])
        # com_for_plot[1].append(action[-1])

        # if done:
        #    plt.figure()
        #    plt.plot(com_for_plot[0],com_for_plot[1],"*")
        #    plt.show()
        #    com_for_plot=[[],[]]
        self.first_step=False
        self.step_counter+=1

        return self.observations, reward, done, {}

    #________________________________________

    def reset(self):
        self.feet_state=False
        self.foot_step_count=0

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
        self.cal_observations(self.last_actions)

        return self.observations
    #________________________________________
    def render(self, mode='human'):
        pass
    def close (self):
        try:
            p.disconnect()
            print(p.getConnectionInfo())
        except: print("_disc e_")
    #________________________________________

#____cast________________________________________________________________________________________
    #from_CAST
    def zmp_ft(self, is_right):
        # measurement of zmp relative to sensor local coordinate using f/t sensor
        sole_id = 5 if is_right else 11
        ft_data = p.getJointState(self.SurenaID, sole_id)[2]
        if ft_data[2] == 0:
            return [0, 0, 0]
        else:
            x_zmp = -(ft_data[4]) / (ft_data[2])
            y_zmp = -(ft_data[3]) / (ft_data[2])
        return [x_zmp, y_zmp, ft_data[2]]

    def cal_ZMP(self): 
        total_zmp = np.zeros((3, 1))
        l_x_zmp, l_y_zmp, l_fz = self.zmp_ft(False)
        l_zmp = self.ankle2pelvis(np.array([l_x_zmp, l_y_zmp, 0.0]), False) # left foot zmp relative to pelvis
        if abs(l_fz) < 5:
            l_fz = 0

        r_x_zmp, r_y_zmp, r_fz = self.zmp_ft(True)
        r_zmp = self.ankle2pelvis(np.array([r_x_zmp, r_y_zmp, 0.0]), True) 
        if abs(r_fz) < 5:
            r_fz = 0
            
        if l_fz + r_fz == 0:
            #print("No foot contact!!")
            pass
        else:
            total_zmp = (r_zmp * r_fz + l_zmp * l_fz) / (l_fz + r_fz)
        return total_zmp

    #________________
   
    def rotateAxisX(self, phi):
        # alpha: angle in rad 
        rot = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        return rot

    def rotateAxisY(self, theta):
        # theta: angle in rad 
        rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        return rot

    def rotateAxisZ(self, psi):
        # psi: angle in rad 
        rot = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        return rot

    def homoTrans(self, axis, q, p):
        # axis: rotation axis relative to global coordinate (0 -> x, 1 -> y, 2 -> z)
        # q: rotation angle
        # p: 3d array indicates translation 
        if axis == 0:
            return(np.block([[self.rotateAxisX(q), p.reshape((3,1))], [np.zeros((1, 3)), 1]]))
        elif axis == 1:
            return(np.block([[self.rotateAxisY(q), p.reshape((3,1))], [np.zeros((1, 3)), 1]]))
        elif axis == 2:
            return(np.block([[self.rotateAxisZ(q), p.reshape((3,1))], [np.zeros((1, 3)), 1]]))

    def ankle2pelvis(self, p_a, is_right):
        # p_a position relative to ankle coordinate
        shank = 0.36
        thigh = 0.37
        torso = 0.115
        if is_right:
            
            t1 = self.homoTrans(2, p.getJointState(self.SurenaID, 0)[0], np.array([0, -0.115, 0])) # hip yaw
            t2 = self.homoTrans(0, p.getJointState(self.SurenaID, 1)[0], np.zeros((1,3))) # hip roll
            t3 = self.homoTrans(1, p.getJointState(self.SurenaID, 2)[0], np.zeros((1,3))) # hip pitch
            t4 = self.homoTrans(1, p.getJointState(self.SurenaID, 3)[0], np.array([0, 0, -thigh])) # knee pitch
            t5 = self.homoTrans(0, 0, np.array([0, 0, -shank]))
            p_w = t1 @ t2 @ t3 @ t4 @ t5 @ (np.block([[p_a.reshape((3, 1))], [1]]))
            return p_w[0:3]

        else:
            t1 = self.homoTrans(2, p.getJointState(self.SurenaID, 6)[0], np.array([0, 0.115, 0])) # hip yaw
            t2 = self.homoTrans(0, p.getJointState(self.SurenaID, 7)[0], np.zeros((1,3))) # hip roll
            t3 = self.homoTrans(1, p.getJointState(self.SurenaID, 8)[0], np.zeros((1,3))) # hip pitch
            t4 = self.homoTrans(1, p.getJointState(self.SurenaID, 9)[0], np.array([0, 0, -thigh])) # knee pitch
            t5 = self.homoTrans(0, 0, np.array([0, 0, -shank]))
            p_w = t1 @ t2 @ t3 @ t4 @ t5 @ (np.block([[p_a.reshape((3, 1))], [1]]))
            return p_w[0:3]

    def is_left(self,P0, P1, P2):
        return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])


    def zmpViolation(self, zmp, V):
        # checks if zmp is inside the polygon shaped by
        # vertexes V using Windings algorithm
        # inspiration: http://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py
        if (V.shape[0] == 8):
            V = np.delete(V,3,0)
            V = np.delete(V,3,0)
            V = V.tolist()
            v = list([V[0],V[2],V[4],V[5],V[3],V[1],V[0]])
            V = np.array(v)
        if (V.shape[0] == 4):
            V = V.tolist()
            v = list([V[0],V[2],V[3],V[1],V[0]])
            V = np.array(v)
            
        wn = 0
        for i in range(len(V)-1):     # edge from V[i] to V[i+1]
            if V[i][1] <= zmp[1]:        # start y <= P[1]
                if V[i+1][1] > zmp[1]:     # an upward crossing
                    if self.is_left(V[i], V[i+1], zmp) > 0: # P left of edge
                        wn += 1           # have a valid up intersect
            else:                      # start y > P[1] (no test needed)
                if V[i+1][1] <= zmp[1]:    # a downward crossing
                    if self.is_left(V[i], V[i+1], zmp) < 0: # P right of edge
                        wn -= 1           # have a valid down intersect
        if wn == 0:           
            return True
        else:
            return False        
    #_____zmp________
    def process_ZMP(self,cnt):

        #getting support polygon
        V = list("")
        ncontact=len(cnt)
        flagR,flagL=True,True
        if ncontact>0:
            for k in range(ncontact):
                if cnt[k][3]==5 and flagR:
                    flagR=False
                    contact = p.getLinkState(self.SurenaID,5)[0]
                    V.append([contact[0]-0.09046, contact[1]-0.0811, contact[0]-0.09046 + contact[1]+0.0811 ])
                    V.append([contact[0]-0.09046, contact[1]+0.0789, contact[0]-0.09046 + contact[1]-0.0789])
                    V.append([contact[0]+0.1746, contact[1]-0.0811, contact[0]+0.1746 + contact[1]+0.0811])
                    V.append([contact[0]+0.1746, contact[1]+0.0789, contact[0]+0.1746 + contact[1]-0.0789])
                if cnt[k][3]==11 and flagL:
                    flagL=False
                    contact = p.getLinkState(self.SurenaID,11)[0]
                    V.append([contact[0]-0.09046, contact[1]+0.0811, contact[0]-0.09046 + contact[1]+0.0811 ])
                    V.append([contact[0]-0.09046, contact[1]-0.0789, contact[0]-0.09046 + contact[1]-0.0789])
                    V.append([contact[0]+0.1746, contact[1]+0.0811, contact[0]+0.1746 + contact[1]+0.0811])
                    V.append([contact[0]+0.1746, contact[1]-0.0789, contact[0]+0.1746 + contact[1]-0.0789])

        zmp_violate=False
        try:
            V = np.array(V)
            V = V[V[:,2].argsort()]
            zmp_violate=self.zmpViolation(self.zmp, V)
        except:
            # print("not enough points")
            zmp_violate=True
            pass

        return not zmp_violate

#________________
    def cal_stepping_reward(self,cforcses):
        delta_s=0.0
        if cforcses[0]>0.001 and cforcses[1]>0.001: #then double support
            self.last_on_floor=self.current_on_floor
        elif cforcses[0]>0.001 and cforcses[1]<0.001: #right on ground
            self.current_on_floor=0
            if self.last_on_floor==1: 
                delta_s=self.current_feet_pos[0][0]-self.current_feet_pos[1][0] #right-left

        elif cforcses[0]<0.001 and cforcses[1]>0.001: #left on ground
            self.current_on_floor=1
            if self.last_on_floor==0: 
                delta_s=self.current_feet_pos[1][0]-self.current_feet_pos[0][0] #left-right
     
        return max(0,delta_s-0.04)

#_____________________


if __name__=="__main__":
    S=SurenaRobot_v1("gui")

    for i in range(500):
        S.step([0]*4)
    # print(S.com)0
    # print("*************",p.getLinkStates(S.SurenaID,S.jointIDs))
    
    # for i in range(10000):
    #     a,b,done,c=S.step([.1,.1,.1,.1])
    #     if done:
    #         S.reset()


    # import json

    # with open('data.txt') as json_file:
    #     data = json.load(json_file)
    #     for pr in data['people']:
    #         com_der=np.reshape(np.array(pr['COM']),(-1,3))
    #         der1=np.reshape(np.array(pr['R']),(-1,3))
    #         der2=np.reshape(np.array(pr['L']),(-1,3))

    # plt.figure()
    # plt.plot(com_der)
    # plt.legend(["x","y","z"])
    # plt.title("der_com")

    # plt.figure()
    # plt.plot(der1)
    # plt.legend(["x","y","z"])
    # plt.title("right der")

    # plt.figure()
    # plt.plot(der2)
    # plt.legend(["x","y","z"])
    # plt.title("left der")

    # plt.show()
    # print(der1)
    # print(der2)

    # for ii in range(21): #2160
        
    #     feet_state=np.abs(S.right_ankle[1]-S.com[1]) > np.abs(S.left_ankle[1]-S.com[1]) #0:right 1:left on the floor
    #     # feet_state=(der2[ii][0]==0)
    #     # if ii<20:print(S.right_ankle[1],S.com[1],S.left_ankle[1],S.com[1],feet_state)
    #     if feet_state:
    #         ac=np.array([com_der[ii][0],com_der[ii][1],der1[ii][0],der1[ii][2]])
    #     else:
    #         ac=np.array([com_der[ii][0],com_der[ii][1],der2[ii][0],der2[ii][2]])
    #     if ii<20:print("obs",S.observations[4:])
    #     for i in range(1000):
    #         S.step(ac)

    print("__end__")
