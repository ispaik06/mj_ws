import gymnasium as gym

env = gym.make("HumanoidStandup-v5")
obs, info = env.reset()
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("OK:", obs.shape)
env.close()
