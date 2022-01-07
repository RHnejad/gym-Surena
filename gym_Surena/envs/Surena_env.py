import gym
from gym import spaces
import pybullet as p
import numpy as np
import time
import pybullet_data

FEEDBACK,FEEDBACK2=[],[]

WITH_GUI = 0

TENDOF=1
MINDOF=0

DELTA_THETA =0
TORQUE_CONTROL=1

ACTIVATION= 1#action
#نکته: بدون اکتیویشن در حالت تورک کنترل نیازه که یه ضریبی رو اضافه کنی که تو اکشن است\ ضرب کنه چون خیلی حروجی های کمی میده
KNEE=0
IMITATE=1
if IMITATE : KNEE = True

NORMALIZE=1#observation

ACTIVATE_SLEEP,A_S_AFTER = 0,None
if not WITH_GUI:
    ACTIVATE_SLEEP=False
if MINDOF:
    TENDOF=0

PLOT_REWS=1
N_plot=10

# file_name="SURENA/sfixedlim.urdf" if not MINDOF else "SURENA/nofootsfixedlim.urdf"
# file_name="SURENA/colorsfixedlimWLes.urdf"
# file_name="SURENA/newFootSfixedlim.urdf"
file_name="SURENA/sfixedlim.urdf"
  
# X0=-0.0517
Z0=0.9727
foot_z0=0.037999 
foot_y0_r=0.11380
T=200.
beta=1.5
gain=1.
num_steps=512


if KNEE:
    from kasra.Robot import *
    from kasra.DCM import *
    surena = Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)
    planner = DCMPlanner(0.7, 1.5, 0.45,1./T) 

class SurenaRobot(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,gui="",frequency=0):
        super(SurenaRobot, self).__init__()

        if PLOT_REWS :self.mean_reward_array=np.zeros((N_plot,12))
        self.episode_num=0 #zero base

        try:
            global WITH_GUI
            if frequency>0:
                global T
                T=frequency
            if gui=="gui":       
                WITH_GUI=1
            elif gui==1:
                WITH_GUI=1
    
        except: print("warning!")

        self.num_actions= (10 if TENDOF else 12) if not MINDOF else 6
        self.observation_dimensions= 2*self.num_actions+14 
        # VelList=[25,25,38,60,50,50,25,25,38,60,50,50]

        self.theta_high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4 ], dtype=np.float32)
        self.theta_low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7], dtype=np.float32)

        self.thetaDot_high=np.multiply (np.array([0.0131,0.0131,0.0199,0.0314,0.0262,0.0262 ,0.0131,0.0131,0.0199,0.0314,0.0262,0.0262], dtype=np.float32),200./T)
        self.thetaDot_low=np.multiply (np.array([-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262 ,-0.0131,-0.0131,-0.0199,-0.0314,-0.0262,-0.0262], dtype=np.float32),200./T)            
        
        self.torques_high=np.multiply(np.array([60,60,40,72,27,27, 60,60,40,72,27*1.5,27*1.5], dtype=np.float32),beta)
        self.torques_low=np.multiply(np.array([-60,-60,-40,-72,-27,-27, -60,-60,-40,-72,-27*1.5,-27*1.5], dtype=np.float32),beta)
        #multiplied to match humanoid_running order and +20 for ankle torques to match better robot versions

        self.obs_high=np.array([0.4,0.1,1.2,2.0,1.3,0.7,   1.0,0.5,1.2,2.0 ,1.3,0.4,
                315,315,315,315,315,315, 315,315,315,315,315,315,
                #60,60,40,72,27,27, 60,60,40,72,27,27,
                10,1.0, 3.0,3.0,3.0, #y z vx vy vz
                1.0,1.0,1.0,1.0, 5.0,5.0,5.0,
                250,250], dtype=np.float32)
        self.obs_low=np.array([-1.0, -0.5,-1.0,0.0,-1.0,-0.7,  -0.4,-0.1,-1.0,0.0,-1.0,-0.7,
                -315,-315,-315,-315,-315,-315, -315,-315,-315,-315,-315,-315,
                #-60,-60,-40,-72,-27,-27,-60,-60,-40,-72,-27,-27,
                -10,0.0, -3.0,-3.0,-3.0,
                -1.0,-1.0,-1.0,-1.0, -5.0,-5.0,-5.0,
                0,0], dtype=np.float32)

        
        if TENDOF:
            self.obs_high,self.obs_low=np.delete(self.obs_high,[0,6,12,18]),np.delete(self.obs_low,[0,6,12,18])
            self.theta_high,self.theta_low=np.delete(self.theta_high,[0,6]),np.delete(self.theta_low,[0,6])
            self.thetaDot_high,self.thetaDot_low=np.delete(self.thetaDot_high,[0,6]),np.delete(self.thetaDot_low,[0,6])
            self.torques_high,self.torques_low=np.delete(self.torques_high,[0,6]),np.delete(self.torques_low,[0,6])
            # VelList=np.delete(VelList,[0,6])
            
        if MINDOF:
            temp=[0,4,5,6,10,11]
            self.theta_high,self.theta_low=np.delete(self.theta_high,temp),np.delete(self.theta_low,temp)
            self.thetaDot_high,self.thetaDot_low=np.delete(self.thetaDot_high,temp),np.delete(self.thetaDot_low,temp)
            self.torques_high,self.torques_low=np.delete(self.torques_high,temp),np.delete(self.torques_low,temp)
            # VelList=np.delete(VelList,temp)
            temp=temp+[12,16,17,18,22,23]
            self.obs_high,self.obs_low=np.delete(self.obs_high,temp),np.delete(self.obs_low,temp)

        #________________________________________
        
        self.observation_space = gym.spaces.box.Box(
                low=self.obs_low,
                high=self.obs_high)

        if ACTIVATION:
            self.action_space=gym.spaces.box.Box(
                low= np.negative(np.ones(self.num_actions,dtype=np.float32)),
                high= np.ones(self.num_actions,dtype=np.float32))
        else:
            if TORQUE_CONTROL:
                self.action_space=(gym.spaces.box.Box(
                        low=self.torques_low,
                        high=self.torques_high)) 
            else:
                self.action_space=(gym.spaces.box.Box(
                        low=self.thetaDot_low,
                        high=self.thetaDot_high)) if DELTA_THETA else (gym.spaces.box.Box(
                        low=self.theta_low,
                    high=self.theta_high))

        self.joint_space = {
                "low":self.theta_low,
                "high":self.theta_high} 

         
        # self.maxVel=np.multiply(np.array(VelList, dtype=np.float32),2*(np.pi)/60)
        # self.minVel=np.multiply(self.maxVel,-1)

        self.observation=np.zeros(self.observation_dimensions,dtype=np.float32)

        self.currentPos=np.zeros((self.num_actions,),dtype=np.float32)
            
        rightLegID=([1,2,3,4,5]  if TENDOF else list(range(6))) if not MINDOF else [1,2,3]
        leftLegID=([7,8,9,10,11]  if TENDOF else list(range(6,12)) ) if not MINDOF else [5,6,7]
        self.jointIDs=rightLegID+leftLegID
            
        self.physicsClient = p.connect(p.GUI) if WITH_GUI else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        self.step_counter=0
        self.first_step=True
        self.up=0
        self.planeId=0
        self.SurenaID=1
        self.zmp=[0,0,0]
        self.startPos = [0,0,0] if not MINDOF else [0,0,-0.075]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
 
        self.current_feet_pos=[]
        self.last_on_floor=None #0:right 1:left
        self.current_on_floor=None

        self.JointStates=None
        self.link_states_feet=None
        self.contacts=None

        self.sum_episode_reward=0

        self.reset()

        if IMITATE:
            import json
            with open('classic200.txt') as json_file:
                data = json.load(json_file)
                for pr in data['robot']:
                    self.des_com=np.reshape(np.array(pr['come']),(-1,3))
                    self.des_right=np.reshape(np.array(pr['right']),(-1,3))
                    self.des_left=np.reshape(np.array(pr['left']),(-1,3))
                    self.des_theta=np.reshape(np.array(pr['theta']),(-1,12))
                
                if TENDOF: 
                    self.des_theta=np.delete(self.des_theta,[0,6],1)



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

    def cal_ZMP_cast(self): 
        total_zmp = np.zeros((3, 1))
        l_x_zmp, l_y_zmp, l_fz = self.zmp_ft(False)
        l_zmp = self.ankle2pelvis(np.array([l_x_zmp, l_y_zmp, 0.0],dtype=np.float32), False) # left foot zmp relative to pelvis
        if abs(l_fz) < 5:
            l_fz = 0

        r_x_zmp, r_y_zmp, r_fz = self.zmp_ft(True)
        r_zmp = self.ankle2pelvis(np.array([r_x_zmp, r_y_zmp, 0.0],dtype=np.float32), True) 
        if abs(r_fz) < 5:
            r_fz = 0
            
        if l_fz + r_fz == 0:
            #print("No foot contact!!")
            pass
        else:
            total_zmp = (r_zmp * r_fz + l_zmp * l_fz) / (l_fz + r_fz)

        # print("CAST ZMP:",total_zmp.T)

        return total_zmp

    #________________
   
    def rotateAxisX(self, phi):
        # alpha: angle in rad 
        rot = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]],dtype=np.float32)
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
            V = np.array(v,dtype=np.float32)
        if (V.shape[0] == 4):
            V = V.tolist()
            v = list([V[0],V[2],V[3],V[1],V[0]])
            V = np.array(v,dtype=np.float32)
            
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

    def cal_footplacement_rew(self):
        if MINDOF:
            return 0
        r_val=0
        for foot in range(2):
            if self.current_feet_pos[foot][2]< 0.05 and self.current_feet_pos[foot][2]> foot_z0:
                delta_or=(p.getEulerFromQuaternion(np.abs(np.array(self.link_states_feet[foot][1])))) #-self.startOrientation
                delta_or=delta_or*np.array([1,1,0.25])
                r_val+=(sum(delta_or)) #/self.current_feet_pos[foot][2]

        if r_val:r_val=0.00005/(r_val)
        return r_val
              
#_____________________
    def cal_ZMP(self,R_z):
        ZMP=np.zeros((2,2))
        mg=1.48*9.81
        for foot in range(2):
            try:
                ZMP[foot][0]=(mg*self.link_states_feet[foot][0][0] + self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][4]-
                        self.link_states_feet[2+foot][0][0]*self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][2]+
                        self.link_states_feet[2+foot][0][2]*self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][0])/R_z[foot]
            except: pass

            try:
                ZMP[foot][1]=(mg*self.link_states_feet[foot][0][1] - self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][3]-
                        self.link_states_feet[2+foot][0][1]*self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][2]+
                    self.link_states_feet[2+foot][0][2]*self.JointStates[5+(6-int(TENDOF))*foot-int(TENDOF)][2][1])/R_z[foot]
            except: pass

            
        # print(ZMP)

        if (R_z[0]+R_z[1])>0:
            x_zmp=(ZMP[0][0]*R_z[0]+ZMP[1][0]*R_z[1])/(R_z[0]+R_z[1])
            y_zmp=(ZMP[0][1]*R_z[0]+ZMP[1][1]*R_z[1])/(R_z[0]+R_z[1])
        else: x_zmp,y_zmp=0.0,0.0

        # print(x_zmp,y_zmp)

        return [x_zmp,y_zmp,0.0]
#obss____________________________________________________________________________________________

    def cal_observations(self): 
        Ts_raw=np.zeros(self.num_actions)
        Theta_dots=np.zeros(self.num_actions)
        SPos, SOrn = p.getBasePositionAndOrientation(self.SurenaID)
        LinearVel,AngularVel=p.getBaseVelocity(self.SurenaID)
        self.JointStates=p.getJointStates(self.SurenaID,self.jointIDs) ##JPos,JVel,JF
        self.contacts=p.getContactPoints(self.SurenaID,self.planeId)

        iscontact=True

        FzR,FzL=0.0,0.0
        iscontact=bool(len(self.contacts))

        rc,lc=0,0
        ncontact=len(self.contacts)
        for k in range(ncontact):
            if self.contacts[k][3]==(5 if not MINDOF else 3):
                FzR+=self.contacts[k][9]
                lc+=1
                
            elif self.contacts[k][3]==(11 if not MINDOF else 7):
                FzL+=self.contacts[k][9]
                rc+=1
                
        # print(FzR,FzL)
        # ZMP_in_SP=self.process_ZMP(self.contacts)
        # ZMP_in_SP=True
        if not MINDOF:
            self.link_states_feet=p.getLinkStates(self.SurenaID,[5,11,4,10]) 
            self.current_feet_pos=[list(self.link_states_feet[0][0]),list(self.link_states_feet[1][0])]
            z_l5,z_l11=self.link_states_feet[0][0][2],self.link_states_feet[1][0][2]

            if z_l5>=0.04 and z_l11>=0.04 :
                iscontact=False
            elif z_l5<=foot_z0 or z_l11<=foot_z0: #double checks and if iscontact was decided as False but in fact ther is contact, corrects it
                iscontact=True
                self.zmp=self.cal_ZMP([FzR,FzL])
            
            # FEEDBACK.append(list(self.zmp[:2]))
            temp=self.cal_ZMP_cast()
            # FEEDBACK2.append(list(self.cal_ZMP_cast))

            if rc==4:
                self.last_feet_floor_x[0]=self.link_states_feet[0][0][0]
            elif lc==4:
                self.last_feet_floor_x[1]=self.link_states_feet[1][0][0]

        #________________________________
        for jj in range(self.num_actions):
            Ts_raw[jj]=self.JointStates[jj][3]
            Theta_dots[jj]=self.JointStates[jj][1]
        Ts=np.absolute(Ts_raw)
        Theta_dots=np.absolute(Theta_dots)
        powers=sum(Ts*Theta_dots)

        x=SPos[0]

        # observation_new=np.zeros(self.observation_dimensions)

        for ii in range(self.num_actions):
            self.observation[ii]=self.JointStates[ii][0] #theta
            self.observation[ii+self.num_actions]=self.JointStates[ii][1] #theta_dot
            
        #without x 
        self.currentPos=self.observation[0:self.num_actions] 

        #observation_new[2*self.num_actions: 3*self.num_actions]=Ts_raw
        self.observation[2*self.num_actions: 2*self.num_actions+2]=np.array(SPos,dtype=np.float32)[1:3]
        self.observation[2*self.num_actions+2: 2*self.num_actions+5]=np.array(LinearVel,dtype=np.float32)

        self.observation[2*self.num_actions+5: 2*self.num_actions+9]=np.array(SOrn,dtype=np.float32)
        self.observation[2*self.num_actions+9: 2*self.num_actions+12]=np.array(AngularVel,dtype=np.float32)

        self.observation[2*self.num_actions+12: 2*self.num_actions+14]=np.array([FzR, FzL],dtype=np.float32) #F_z_r and F_z_l

        return iscontact, powers, x#, ZMP_in_SP

    #________________________________________
    def cal_reward(self,powers,x):
        sum_thetaDot=sum((self.observation[self.num_actions:2*self.num_actions])**2)
        sum_orn=sum(np.abs(self.startOrientation-self.observation[2*self.num_actions+5: 2*self.num_actions+9]))
        stepping_reward=self.cal_stepping_reward(self.observation[-2:])
        imitation_reward=0
        if IMITATE:
            imitation_reward=np.power((self.observation[0:self.num_actions]-self.des_theta[self.step_counter%1800]),2)
            imitation_reward=np.sum(imitation_reward)
            imitation_reward=np.exp(-1*imitation_reward) #chenge -1 another negative num. if necessary


        #[0.x,1.x_dot,2.stepCount,  3.done,4.power,5.dy,6.dz 7.SigmathethatDot^2 8.SigmaabsdeltaOrn]
        param=np.array([x, 
            powers, 
            max(0,  np.exp( np.abs(self.observation[2*self.num_actions])-0.130)  -1.), 
            max(0,  np.exp( np.abs(self.observation[2*self.num_actions+1]-Z0)-0.03)  -1.), 
            (self.step_counter/num_steps),
            sum_thetaDot,
            sum_orn,
            stepping_reward,
            min(  max(self.current_feet_pos[0][0],self.current_feet_pos[1][0]) -x   ,0),
            min(self.observation[2*self.num_actions+2]-0.12,0), # v_x
            self.cal_footplacement_rew(),
            imitation_reward])

        # weights=np.array([ 2.3 , 0.0 ,-0.3 ,0.0, 0 ,0, -1.7, +0.0, 0.0,  0.0, 0.08, 0.7],dtype=np.float32)
        weights=np.array([ 2.5, 0.0 ,0 ,0.0, 0 ,0, -0.05, +0.0, 0.0,  0.0, 0.7, 0.9],dtype=np.float32)

        #heree
        reward_array=param*weights
        reward_s=sum(reward_array)+0.65 #-0.75* self.up#-0.095 #-0.007*float(bool(self.up))
        reward_s=reward_s/4   
        # print(reward_array)

        if PLOT_REWS : self.mean_reward_array[self.episode_num%N_plot]+=param

        return reward_s


    #________________________________________

    def step(self, action):
        if ACTIVATE_SLEEP:
            time.sleep(1./T)
        
        # print("raw",action)

        if ACTIVATION:
            a=self.torques_high-self.torques_low if TORQUE_CONTROL else (self.thetaDot_high-self.thetaDot_low  if DELTA_THETA else self.theta_high-self.theta_low)
            b=self.torques_high+self.torques_low if TORQUE_CONTROL else (self.thetaDot_high+self.thetaDot_low  if DELTA_THETA else self.theta_high+self.theta_low )
            # action=(np.divide((self.action_space.high-self.action_space.low),2))*action
            action=(np.divide(a,2))*action +(np.divide(b,2))

            
        action=action+self.currentPos if DELTA_THETA else action
        if TORQUE_CONTROL:
            action*=gain
            # print(action)
            p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                            jointIndices=self.jointIDs,
                            controlMode=p.TORQUE_CONTROL,
                            forces= action) 

        else:
            for i in range(self.num_actions):
                if action[i]<self.joint_space["low"][i]:
                    action[i]=self.joint_space["low"][i]
                elif action[i]>self.joint_space["high"][i]:
                    action[i]=self.joint_space["high"][i]
            p.setJointMotorControlArray(bodyUniqueId=self.SurenaID,
                                        jointIndices=self.jointIDs,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions = action)  #forces=np.full((self.num_actions,),600)

        p.stepSimulation()
            
        iscontact, powers, x=self.cal_observations() 
        # pace_penalty=cal_pace_penalty(0) #for right foot
        # pace_penalty+=cal_pace_penalty(1) #for left foot
        if not iscontact:
            self.up+=1
        else:
            self.up=0
    
        done=(self.observation[2*self.num_actions+1]<(0.6)) #or (self.up>20) #or .... ????  #or (not ZMP_in_SP)
        if MINDOF: done = done or (self.observation[2*self.num_actions+1]>(0.9))

        reward=self.cal_reward(powers,x)

        print(self.mean_reward_array)
        
        if not done: self.step_counter+=1   
        elif PLOT_REWS : 
            self.mean_reward_array[self.episode_num%N_plot]= np.divide(self.mean_reward_array[self.episode_num%N_plot],self.step_counter+1)
            self.episode_num+=1  
        if done and  self.episode_num%N_plot==0 and self.episode_num>0 and PLOT_REWS :
            fig=plt.figure()
            plt.plot(self.mean_reward_array)
            plt.show()
            plt.close()

            
        # if done: print(self.step_counter)
        self.first_step=False

        if NORMALIZE:
            temp=self.obs_high-self.obs_low
            self.observation=self.observation/temp
            
        return self.observation, reward, done, {}

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
        if ACTIVATE_SLEEP:
            time.sleep(1)

    #________________________________________

    def reset(self):
        self.first_step=True
        self.step_counter=0
        p.resetSimulation()
        self.planeId = p.loadURDF("plane.urdf") #ID:0
        self.SurenaID=p.loadURDF(file_name,self.startPos, self.startOrientation,useFixedBase=False) 
        p.enableJointForceTorqueSensor(self.SurenaID,5)
        p.enableJointForceTorqueSensor(self.SurenaID,11)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1./T)

        if KNEE:
            self.bend_knee()

        # p.stepSimulation()
        self.current_feet_pos=[[0.0054999,foot_y0_r,foot_z0],[0.0054999,-1*foot_y0_r,foot_z0]]
        self.last_feet_floor_x=[0.0054999,0.0054999]
        self.pace_cout=[0,0]
        
        self.cal_observations()[0]
        if NORMALIZE:
            temp=self.obs_high-self.obs_low
            self.observation=self.observation/temp

        if TORQUE_CONTROL:
            for j in self.jointIDs:
                if p.getJointInfo(self.SurenaID, j)[2] == p.JOINT_REVOLUTE: 
                    p.setJointMotorControl2(self.SurenaID, j, controlMode=p.VELOCITY_CONTROL, force=0)
        
        return self.observation

    #________________________________________
    def render(self, mode='human'):
        pass
    def close (self):
        p.disconnect()
        print(p.getConnectionInfo())
    #________________________________________







if __name__=="__main__":
    # import json

    # with open('classic+rl/data.txt') as json_file:
    #     data = json.load(json_file)
    #     for pr in data['people']:
    #         Torqs=np.reshape(np.array(pr['Torques']),(-1,12))
    #         ac2=np.reshape(np.array(pr["ttheta"]),(-1,12))         

    S=SurenaRobot("gui")

    for i in range(216000): #25920
        time.sleep(1/T)
    #     #S.step(Torqs[i])
        S.step([0]*12)
    #     S.step(ac2[i])

    # fb=np.reshape(np.array(FEEDBACK),(-1,2))
    # fb2=fb.T
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(fb2[0],fb2[1],"*")

    #plt.figure()
    # plt.plot(np.reshape(np.array(FEEDBACK,dtype=np.float32),(-1,12)))
    # plt.figure()
    # plt.plot(np.reshape(np.array(FEEDBACK2,dtype=np.float32),(-1,12)))
    # plt.figure()
    # plt.plot(Torqs)
    # plt.legend(list(range(12)))
    # plt.show()

    print("__done__")




#____step placement___________________________________
    # def create_step_plcament(self):
    #     pass
    #     if right_foot_first :
    #         if self.foot_place_count_right==0:
    #             next_foot_place_right=0.5*self.dS_right*self.foot_place_count_right+1
    #     elif self.foot_place_count_left==0:
    #             next_foot_place_left=0.5*self.dS_left*self.foot_place_count_left+1
    #     else:        
    #         next_foot_place_right=self.dS_right*self.foot_place_count_right+1
    #         next_foot_place_left=self.dS_left*self.foot_place_count_left+1

    # def cal_pace_penalty(self,foot):
    #     deltaX=self.current_feet_pos[foot][0]-self.last_feet_floor_x[foot]
    #     if deltaX<0:
    #         return 0 #CHECK, CHANGE I DON'T KNOW !!!!
    #     if ?
    #     theta_limit=np.arcsin(np.sqrt((self.paces[foot]-deltaX)/self.paces[foot]))
    #     theta_current=np.arctan()

    #     pass
