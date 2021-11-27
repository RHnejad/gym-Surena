# import gym
from stable_baselines3 import PPO

from Surena_env import *
from env_util import make_vec_env,SubprocVecEnv,DummyVecEnv


from multiprocessing import Process


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed



#bash command for tensorboard: "tensorboard --logdir ./tensorboard_logs/ "
#for other info about plotting (and logginf images) visit: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-figures-plots

TRAIN=1
PREV=20
name="TestBeta1orinwithactivation_env"
num_cpu = 1


import sys
custom_objects = {}
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }


def make_env(env_id, rank, seed=0):
    def _init():
        temp="" #new
        if num_cpu==1: temp=env_id #new
        env = SurenaRobot(temp)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init



def main():
    
    start_time = time.time()
    # env=make_vec_env("guei", n_envs=1)#SurenaRobot("guui") #gym.make("Pendulum-v0")
    env = SubprocVecEnv([make_env("gui", i) for i in range(num_cpu)])


    if TRAIN:
        # env =make_vec_env("gui", n_envs=1)#,vec_env_cls=DummyVecEnv)#vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None
        if PREV:           
            model=PPO.load("./savedmodels/"+name+str(PREV-1),env=env,custom_objects=custom_objects, tensorboard_log="./tensorboard_logs/")
        else:
            model = PPO("MlpPolicy", env, verbose=1,n_steps=512,batch_size=16, learning_rate = 3e-4,tensorboard_log="./tensorboard_logs/")
            
        for i in range(PREV,33):
            print("________"+str(i)+"________")
            model.learn(total_timesteps=100000, tb_log_name=name+str(i),reset_num_timesteps=True if (i==PREV and i!=0) else False )  
            #Ulearning_rate: Union[float, Schedule] = 3e-4 
            model.save("./savedmodels/"+name+str(i)) #2000000    

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

if __name__ == '__main__':
    main()
    # # p = Process(target=main)
    # # p.start()
    # # p.join()
    # processes=[Process(target=main) for i in range(2)]
    # [p.start() for p in processes]
    # [p.join() for p in processes]
