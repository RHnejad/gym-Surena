import gym
from stable_baselines3 import PPO
from Surena_v1_env import *

#bash command for tensorboard: "tensorboard --logdir ./tensorboard_logs/ "
#for other info about plotting (and logginf images) visit: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-figures-plots



TRAIN=1
PREV=2


import sys
custom_objects = {}
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

start_time = time.time()
env=SurenaRobot_v1("gui") #gym.make("Pendulum-v0")
if TRAIN:    
    if PREV:           
        model=PPO.load("./colab/colabnov16"+str(PREV-1),env=env, tensorboard_log="./tensorboard_logs/",custom_objects=custom_objects) 
    else:
        model = PPO("MlpPolicy", env, verbose=1,n_steps=512,batch_size=16, tensorboard_log="./tensorboard_logs/")
               
    for i in range(PREV,4):
        print("________"+str(i)+"________")
        model.learn(total_timesteps=100000, tb_log_name="nov16"+str(i)) #,reset_num_timesteps=False      
        model.save("./colab/nov16"+str(i)) #2000000
    
else:
    model=PPO.load("./colab/colab10dof21",env=env,custom_objects=custom_objects)
    obs = env.reset()
    for i in range(1000000):
        action, _states = model.predict(obs, deterministic=0)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()

env.close()
print("--- %s seconds ---" % (time.time() - start_time))
print("___done___")
