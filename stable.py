# import gym
import torch

from stable_baselines3 import PPO,TD3

from Surena_env import *
from env_util import make_vec_env,SubprocVecEnv,DummyVecEnv

from multiprocessing import Process

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback

#bash command for tensorboard: "tensorboard --logdir ./tensorboard_logs/ "
#for other info about plotting (and logginf images) visit: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-figures-plots

#
#jan9thTORQonlyX_600000_steps.zip
def main():
    TRAIN=True
    PREV=1
    N=20
    name="jan16withIMITdeltaTheta"
    num_cpu =1
    CtlCSave=0
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[64,64,32], vf=[64,64,32])])

    checkpoint_callback = CheckpointCallback(save_freq=200000, save_path='./savedmodels/',name_prefix=name)
    start_time = time.time()
    env=make_vec_env("gui", n_envs=num_cpu,vec_env_cls=SubprocVecEnv)#SurenaRobot("guui") #gym.make("Pendulum-v0")
    # env = SubprocVecEnv([make_env("gui", i) for i in range(num_cpu)])
    # env = SurenaRobot("gui")

    if TRAIN:
        if PREV: model=PPO.load("./savedmodels/"+name+"last",env=env, custom_objects=custom_objects,tensorboard_log="./tensorboard_logs/")   #     
        else: model = PPO("MlpPolicy", env, verbose=1,n_steps=512,batch_size=32, n_epochs=15, learning_rate = 1.5e-4,tensorboard_log="./tensorboard_logs/",policy_kwargs=policy_kwargs)

        try:
            model.learn(total_timesteps=N*100000, tb_log_name=name, log_interval=1,  callback=checkpoint_callback)#, reset_num_timesteps=True if (i==PREV and i!=0) else False)
            model.save("./savedmodels/"+name+"last")
        except KeyboardInterrupt:
            if CtlCSave: model.save("./savedmodels/"+name+"keyInterupts"+str(int(time.time()/1e6))) 



    else:
        env=SurenaRobot()
        model=PPO.load("./colab/1/colab10dof21",env=env)#,custom_objects=custom_objects)
        obs = env.reset()
        for i in range(1000000):
            action, _states = model.predict(obs, deterministic=0)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                obs = env.reset()

    env.close()

    print("--- %s mins ---" % ((time.time() - start_time)/60))


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True
#https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

import sys
custom_objects = {}
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
if newer_python_version:
    custom_objects = {
        "learning_rate": 2e-4 ,
        "lr_schedule": lambda _:2e-4,
        "clip_range": lambda _: 0.2,
    }

def make_env(env_id, rank, seed=0):
    def _init():
        # temp="" #new
        # if num_cpu==1: temp=env_id #new
        env = SurenaRobot(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    main()
    print("___done___")












class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True
        # Log scalar value (here a random variable)
        value = np.random.random()
        summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True








#______________________________________________________________________-
def foremer_main():
    num_cpu=1
    start_time = time.time()
    # env=make_vec_env("guei", n_envs=1)#SurenaRobot("guui") #gym.make("Pendulum-v0")
    env = SubprocVecEnv([make_env("gsui", i) for i in range(num_cpu)])
    # env = SurenaRobot("gui")
    TRAIN,PREV=1,0
    name="old"
    if TRAIN:
        # env =make_vec_env("gui", n_envs=1)#,vec_env_cls=DummyVecEnv)#vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None
        if PREV: model=PPO.load("./savedmodels/"+name+str(PREV-1),env=env,custom_objects=custom_objects, tensorboard_log="./tensorboard_logs/")
        else: model = PPO("MlpPolicy", env, verbose=1,n_steps=512,batch_size=16, learning_rate = 3e-4,tensorboard_log="./tensorboard_logs/")
       
        for i in range(PREV,25):
            print("________"+str(i)+"________")
            model.learn(total_timesteps=100000, tb_log_name=name+str(i),reset_num_timesteps=True if (i==PREV and i!=0) else False )  
            #Ulearning_rate: Union[float, Schedule] = 3e-4 

    else:
        env=SurenaRobot()
        model=PPO.load("./colab/1/colab10dof21",env=env,custom_objects=custom_objects)
        obs = env.reset()
        for i in range(1000000):
            action, _states = model.predict(obs, deterministic=0)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                obs = env.reset()

    env.close()

    print("--- %s mins ---" % ((time.time() - start_time)/60))
    print("___done___")

#model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
#model.learn(total_timesteps=10000, log_interval=10)
#env = model.get_env()


    # # p = Process(target=main)
    # # p.start()
    # # p.join()
    # processes=[Process(target=main) for i in range(2)]
    # [p.start() for p in processes]
    # [p.join() for p in processes]

def main_TD3():
    TRAIN=True
    PREV=True
    N=5
    name="td3colab"
    start_time = time.time()
    env = SurenaRobot("gui") #td3colab_300000_steps

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./savedmodels/',
                                         name_prefix=name)
    if TRAIN:
        if PREV: model=TD3.load("./savedmodels/"+name+"last",env=env, tensorboard_log="./tensorboard_logs/")   #custom_objects=custom_objects,     
        else: model = TD3("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard_logs/") #n_steps=512,batch_size=16, learning_rate = 3e-4,
       
        model.learn(total_timesteps=N*100000, tb_log_name=name, callback=checkpoint_callback , reset_num_timesteps=True if (i==PREV and i!=0) else False)
        model.save("./savedmodels/"+name+"last")

    else:
        model=TD3.load("./colab/1/colab10dof21",env=env)#,custom_objects=custom_objects)
        obs = env.reset()
        for i in range(1000000):
            action, _states = model.predict(obs, deterministic=0)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                obs = env.reset()

    env.close()


def main_TD3():
    TRAIN=1
    PREV=0
    name="dntd3"
    env=SurenaRobot("guis")


    if TRAIN:
        if 11: model=TD3.load(name+str(PREV-1),env=env,tensorboard_log="./tensorboard_logs/",custom_objects=custom_objects)
        else: model=TD3("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard_logs/")
            
        for i in range(11,15):
            print("________"+str(i)+"________")
            model.learn(total_timesteps=100000, tb_log_name=name+str(i))#,reset_num_timesteps=True if (i==PREV and i!=0) else False )  
            #Ulearning_rate: Union[float, Schedule] = 3e-4 
            model.save("./savedmodels/"+name+str(i)) 

    else:
        env=SurenaRobot()
        model=PPO.load("./colab/1/colab10dof21",env=env,custom_objects=custom_objects)
        obs = env.reset()
        for i in range(1000000):
            action, _states = model.predict(obs, deterministic=0)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                obs = env.reset()

    env.close()


