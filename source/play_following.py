import os
import re
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Humanoid-v5"

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")  # 20251226_171745 형태


def latest_run_dir(logs_dir: str = "logs") -> str:
    candidates = [
        d for d in os.listdir(logs_dir)
        if RUN_DIR_RE.match(d) and os.path.isdir(os.path.join(logs_dir, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories like YYYYMMDD_HHMMSS under: {logs_dir}")
    return os.path.join(logs_dir, max(candidates))


def main():
    run_dir = latest_run_dir("logs")
    model_path = os.path.join(run_dir, "checkpoints", "final_model.zip")
    vec_path = os.path.join(run_dir, "checkpoints", "vecnormalize.pkl")

    print("[run ]", run_dir)
    print("[load]", model_path)
    print("[vec ]", vec_path)

    # model = PPO.load(model_path, device="cpu")  # 플레이는 CPU 권장
    model = PPO.load(model_path)

    def make_env():
        # camera_name="track"이 먹으면 그대로 추적, 안 먹어도 아래 lookat으로 해결됨
        return gym.make(ENV_ID, render_mode="human", camera_name="track")

    base_env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, base_env)
    env.training = False
    env.norm_reward = False

    # ✅ reset 먼저!
    obs = env.reset()

    # ✅ VecNormalize/DummyVecEnv 내부의 실제 gym env
    gym_env = env.venv.envs[0]

    # ✅ 이제 render 호출 가능 → viewer 생성
    gym_env.render()

    viewer = gym_env.unwrapped.mujoco_renderer.viewer
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20

    while True:
        # 로봇 위치를 바라보게 업데이트
        x, y, z = gym_env.unwrapped.data.qpos[0:3]
        viewer.cam.lookat[:] = np.array([x, y, z])

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()


if __name__ == "__main__":
    main()