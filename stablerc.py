import gym
from stable_baselines3 import PPO
from rlclassic import *

#bash command for tensorboard: tensorboard --logdir ./tensorboard_logs/
#for other info about plotting (and logginf images) visit: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-figures-plots

TRAIN=1
CONTINUE_FROM_PREV=1

if TRAIN:
  env=SurenaRobot_v1() #gym.make("Pendulum-v0")
  if CONTINUE_FROM_PREV: 
      model=PPO.load("v1_attempt3",env=env, tensorboard_log="./tensorboard_logs/") 
  else:
      model = PPO("MlpPolicy", env, verbose=1,n_steps=2048*50,batch_size=16, tensorboard_log="./tensorboard_logs/")
  model.learn(total_timesteps=400000, tb_log_name="first_run_log",reset_num_timesteps=False)
  model.save("v1_attempt4") #2000000

else:
  env=SurenaRobot_v1("gui")
  model=PPO.load("v1_attempt3",env=env)
  obs = env.reset()
  for i in range(100000):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()

env.close()
print("___done___")
