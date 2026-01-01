import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def find_latest_run(logs_dir="logs"):
    run_dirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir)
                if os.path.isdir(os.path.join(logs_dir, d))]
    latest_run = max(run_dirs, key=os.path.getmtime)
    return latest_run


latest_run = find_latest_run("logs")
model_path = os.path.join(latest_run, "checkpoints", "final_model.zip")
vec_path   = os.path.join(latest_run, "vecnormalize.pkl")

print("[load]", model_path)
print("[vec ]", vec_path)

model = PPO.load(model_path)

# ✅ render 가능한 env를 VecEnv로 감싼 뒤 VecNormalize 로드
base_env = DummyVecEnv([lambda: gym.make("HumanoidStandup-v5", render_mode="human")])
env = VecNormalize.load(vec_path, base_env)

env.training = False      # 평가 모드
env.norm_reward = False   # reward 정규화는 보통 끔(보기/로그용)

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
