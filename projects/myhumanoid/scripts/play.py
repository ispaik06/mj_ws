import os
import re

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ✅ 프로젝트 생성 시 자동 치환
ENV_ID = "Humanoid-v5"

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")  # YYYYMMDD_HHMMSS


def env_tag(env_id: str) -> str:
    """Make a filesystem-safe tag from ENV_ID."""
    return env_id.lower().replace("-", "_")


def latest_run_dir(logs_root: str = "logs", env_id: str = ENV_ID) -> str:
    """
    Return the most recent run directory under:
      logs/<env_tag>/<YYYYMMDD_HHMMSS>
    """
    base = os.path.join(logs_root, env_tag(env_id))
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Env log directory not found: {base}")

    candidates = [
        d for d in os.listdir(base)
        if RUN_DIR_RE.match(d) and os.path.isdir(os.path.join(base, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories like YYYYMMDD_HHMMSS under: {base}")

    latest = max(candidates)  # lexicographic max works with YYYYMMDD_HHMMSS
    return os.path.join(base, latest)


def main():
    # ✅ 커스텀 env 등록(프로젝트 env_id) 보장
    import envs  # noqa: F401

    run_dir = latest_run_dir("logs", ENV_ID)

    model_path = os.path.join(run_dir, "checkpoints", "final_model.zip")
    vec_path = os.path.join(run_dir, "checkpoints", "vecnormalize.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {vec_path}")

    print("[run ]", run_dir)
    print("[load]", model_path)
    print("[vec ]", vec_path)

    model = PPO.load(model_path)

    # Evaluation env with rendering
    base_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human", camera_name="track")])

    # Load VecNormalize stats and freeze them
    env = VecNormalize.load(vec_path, base_env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if bool(done[0]):
            obs = env.reset()


if __name__ == "__main__":
    main()
