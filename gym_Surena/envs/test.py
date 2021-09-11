from Surena_env import SurenaRobot
import gym

myrobot=SurenaRobot()

for i in range(1000):
	myrobot.step([1,1,1,1,1,1,1,1,1,1])
	
myrobot.reset()


for i in range(1000):
	myrobot.step([0,0,0,0,0,0,0,0,0,0])


