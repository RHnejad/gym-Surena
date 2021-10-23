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
MINDOF=1
DELTA_THETA =0
ACTIVATION=0
KNEE=0
NORMALIZE=1
WITH_GUI =1
ACTIVATE_SLEEP,A_S_AFTER = False,None
SAVE_MODEL=0

if not WITH_GUI:
    ACTIVATE_SLEEP=False
if MINDOF:
    TENDOF=0


#file_name="/content/gym-Surena/gym_Surena/envs/SURENA/sfixed.urdf"#google_colab_!git clone https://github.com/RHnejad/gym-Surena.git
file_name="SURENA/sfixed.urdf" if not MINDOF else "SURENA/sfixedMIN.urdf"
#file_name="SURENA/sfixed.urdf" if TENDOF else "SURENA/sfixed12.urdf" 

    
X0=-0.517
Z0=0.9727
foot_z0=0.03799
T=50.

if KNEE:
    from kasra.Robot import *
    from kasra.DCM import *
    surena = Robot(shank = 0.36, hip = 0.37, pelvis_lengt = 0.115)
    planner = DCMPlanner(0.7, 1.5, 0.45,1./T) 


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
            self.obs_high,self.obs_low=np.delete(self.obs_high,[0,6,12,18,24,30]),np.delete(self.obs_low,[0,6,12,18,24,30])
            self.theta_high,self.theta_low=np.delete(self.theta_high,[0,6]),np.delete(self.theta_low,[0,6])
            self.thetaDot_high,self.thetaDot_low=np.delete(self.thetaDot_high,[0,6]),np.delete(self.thetaDot_low,[0,6])
            VelList=np.delete(VelList,[0,6])

        if MINDOF:
            temp=[0,1,4,5,6,7,10,11, 12,13,16,17,18,19,22,23, 24,25,28,29,30,31,34,35]
            self.obs_high,self.obs_low=np.delete(self.obs_high,temp),np.delete(self.obs_low,temp)
            self.theta_high,self.theta_low=np.delete(self.theta_high,[0,1,4,5,6,7,10,11]),np.delete(self.theta_low,[0,1,4,5,6,7,10,11])
            self.thetaDot_high,self.thetaDot_low=np.delete(self.thetaDot_high,[0,1,4,5,6,7,10,11]),np.delete(self.thetaDot_low,[0,1,4,5,6,7,10,11])
            VelList=np.delete(VelList,[0,1,4,5,6,7,10,11])


        #________________________________________
        
        self.observation_space = gym.spaces.box.Box(
                low=self.obs_low,
                high=self.obs_high)

        self.action_space=(gym.spaces.box.Box(
                low=self.thetaDot_low,
                high=self.thetaDot_high)) if DELTA_THETA else (gym.spaces.box.Box(
                low=self.theta_low,
                high=self.theta_high))

        self.joint_space = {
                "low":self.theta_low,
                "high":self.theta_high} 

         
        self.maxVel=np.multiply(np.array(VelList, dtype=np.float32),2*(np.pi)/60)
        self.minVel=np.multiply(self.maxVel,-1)

        self.num_actions= (10 if TENDOF else 12) if not MINDOF else 4
        self.observation_dimensions= 3*self.num_actions+14 

        self.currentPos=np.zeros((self.num_actions,))
            
        self.rightLegID=([1,2,3,4,5]  if TENDOF else list(range(6))) if not MINDOF else [2,3]
        self.leftLegID=([7,8,9,10,11]  if TENDOF else list(range(6,12)) ) if not MINDOF else [8,9]
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

        if ACTIVATE_SLEEP:
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
        if ACTIVATE_SLEEP:
            time.sleep(1)

        

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




#https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)

# #new SurenaRobot-v0
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PPO agent')
#     # Common arguments
exp_name="test1clearrl.py"
gym_id="gym_Surena:SurenaRobot" #HopperBulletEnv-v0"
learning_rate=1e-3 #3e-4
seed=1
total_timesteps= 1000000#2000000
torch_deterministic=True
cuda=True
track=0
capture_video=0#True
wandb_project_name="cleanRL"
wandb_entity=None

#HERE

# Algorithm specific arguments
n_minibatch=32
num_envs=1
num_steps=1024#2048
gamma=0.99
gae_lambda=0.95
ent_coef=0.0
vf_coef=0.5
max_grad_norm=0.5
clip_coef=0.2
update_epochs=10
kle_stop=False
kle_rollback=False
target_kl=0.03
gae=True
norm_adv=True
anneal_lr=True
clip_vloss=True

#args = parser.parse_args()
    # if not seed:
    #     seed = int(time.time())

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // n_minibatch)
                     

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        if ACTIVATION:
            action = np.clip(action,-1, 1)     #new
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high) 
        
        return self.env.step(action)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{gym_id}__{exp_name}__{seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
# writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
#         '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))) new-> comented
if track:
    import wandb
    wandb.init(project=wandb_project_name, entity=wandb_entity, sync_tensorboard=True,  name=experiment_name, monitor_gym=True, save_code=True) #config=vars(args),
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
def make_env(gym_id, seed, idx):
    def thunk():
        env = SurenaRobot()#gym.make(gym_id) new
        env = ClipActionsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = Monitor(env, f'videos/{experiment_name}')
        env = NormalizedEnv(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
envs = VecPyTorch(DummyVecEnv([make_env(gym_id, seed+i, i) for i in range(num_envs)]), device)
# if track:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(gym_id, seed+i, i) for i in range(num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, Box), "only continuous action space is supported"

# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__() #new both networks used to be 1 hiddenlayer (3layers )with 64 nuerons
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256,512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(envs.action_space.shape)), std=0.01),
            #nn.Tanh()         
        ) #new
        self.actor_logstd = nn.Parameter(torch.full((1, np.prod(envs.action_space.shape)),-1.0)) #new torch.zeros(1, np.prod(envs.action_space.shape))
        

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_value(self, x):
        return self.critic(x)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
if anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * learning_rate

# ALGO Logic: Storage for epoch data
obs = torch.zeros((num_steps, num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(num_envs).to(device)
num_updates = total_timesteps // batch_size
for update in range(1, num_updates+1):

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",update)
    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            action, logproba, _ = agent.get_action(obs[step])

        actions[step] = action
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
                break

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,)+envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,)+envs.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizaing the policy and value network
    target_agent = Agent(envs).to(device)
    inds = np.arange(batch_size,)
    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)
        target_agent.load_state_dict(agent.state_dict())
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if kle_stop:
            if approx_kl > target_kl:
                break
        if kle_rollback:
            if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])[1]).mean() > target_kl:
                agent.load_state_dict(target_agent.state_dict())
                break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if kle_stop or kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)


    # if update==10:
    #   print(ACTION)
    #   break

    if SAVE_MODEL:
        if update%100 == 1 or update==num_updates:
            print("*")
            print(agent.actor_mean)
            torch.save(agent.actor_mean,"./modelTest")
            print("saved model", update)

    if update==A_S_AFTER:
        ACTIVATE_SLEEP=True
        

with open('actions.txt', 'w') as filehandle:
    for listitem in ACTION:
        filehandle.write('%s,\n' %listitem)

# envs.close()
writer.close()

print("**********FINISHED*************")

#f011a0c1860c1c1e3d09e05bf66c2dadcb7dd61e
