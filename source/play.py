import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

ENV_ID = "Humanoid-v5"


import re
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")  # 20251226_171745 형태

def latest_run_dir(logs_dir: str = "logs") -> str:
    candidates = [
        d for d in os.listdir(logs_dir)
        if RUN_DIR_RE.match(d) and os.path.isdir(os.path.join(logs_dir, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories like YYYYMMDD_HHMMSS under: {logs_dir}")

    latest = max(candidates)  # 이름이 곧 시간이라 문자열 max가 최신
    return os.path.join(logs_dir, latest)


def main():
    run_dir = latest_run_dir("logs")
    model_path = os.path.join(run_dir, "checkpoints", "final_model.zip")
    vec_path = os.path.join(run_dir, "checkpoints", "vecnormalize.pkl")

    print("[run ]", run_dir)
    print("[load]", model_path)
    print("[vec ]", vec_path)

    model = PPO.load(model_path)

    base_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human", camera_name="track")])
    # base = gym.make(ENV_ID, render_mode="human")
    # base = TimeLimit(base.unwrapped, max_episode_steps=1000)  # 원하는 값
    # base_env = DummyVecEnv([lambda: base])

    env = VecNormalize.load(vec_path, base_env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()


if __name__ == "__main__":
    main()