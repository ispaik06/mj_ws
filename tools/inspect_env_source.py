import gymnasium as gym
import inspect, gymnasium.envs.mujoco as mujoco_envs

env = gym.make("HumanoidStandup-v5")
print(type(env.unwrapped))
print(inspect.getsourcefile(type(env.unwrapped)))
