test / in progress

" pip install -e . " (in gym-Surena) After you have installed your package with pip install -e gym-foo, you can create an instance of the environment with gym.make('gym_foo:foo-v0')

SURENA folder (which includes the .urdf model) should be in the code's folder.
Surena_Robot_v1 is a hybrid of RL and classic methods. Trajectories for ankles and COM will be found by actor-network in each time_step (DRL) and robot joint positions will be calculated by inverse kinematics (classic)

Surena_v0
![image](https://user-images.githubusercontent.com/30596230/150094784-f112324a-0aa5-496b-88c2-e53910b8ce61.png)

Surena_v1
![image](https://user-images.githubusercontent.com/30596230/150094730-2e5b3c2f-ddf0-49cc-8d8c-481716529181.png)
