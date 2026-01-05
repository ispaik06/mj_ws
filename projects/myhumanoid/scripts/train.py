import os
from datetime import datetime
from multiprocessing import cpu_count

import gymnasium as gym
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ✅ 프로젝트 생성 시 자동 치환
ENV_ID = "myhumanoid-v0"

# ✅ 메인 프로세스에서도 등록 한번 실행
import envs  # noqa: F401


def make_env(rank: int, seed: int = 0):
    """Factory function for creating a single environment instance for vectorized training."""
    def _init():
        # ✅ spawn 워커에서도 등록이 되도록(중요)
        import envs  # noqa: F401
        env = gym.make(ENV_ID)
        env = Monitor(env)  # Records episode reward/length to support monitoring and logging.
        env.reset(seed=seed + rank)
        return env
    return _init


def steps_tag(t: int) -> str:
    """Return a fixed-width step tag for consistent filename sorting."""
    return f"{t:07d}"


def video_filename(env_id: str, t: int) -> str:
    """Build a standardized video filename based on environment id and checkpoint steps."""
    env_tag = env_id.lower().replace("-", "_")
    return f"ppo_{env_tag}_steps_{steps_tag(t)}.mp4"


def record_video(model, vecnorm_path: str, out_mp4: str, max_steps: int = 2000, fps: int = 30):
    """Run one rollout with the given model and write an MP4 video to disk."""
    base_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

    # Apply VecNormalize statistics if available to ensure evaluation uses training-time normalization.
    if vecnorm_path and os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, base_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = base_env

    obs = eval_env.reset()

    out_dir = os.path.dirname(out_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with imageio.get_writer(out_mp4, fps=fps) as writer:
        for _ in range(max_steps):
            frame = base_env.envs[0].render()
            if frame is not None:
                writer.append_data(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)

            if bool(dones[0]):
                break

    eval_env.close()
    print(f"[video] saved: {out_mp4}")


class CheckpointAndVideoCallback(BaseCallback):
    """Callback to save model checkpoints and rollout videos at specified timesteps."""
    def __init__(self, checkpoints, run_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoints = sorted(checkpoints)
        self.run_dir = run_dir
        self.next_idx = 0

        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.video_dir = os.path.join(run_dir, "videos")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def _save_vecnormalize(self) -> str:
        """Persist VecNormalize statistics for consistent evaluation and future loading."""
        vec_path = os.path.join(self.ckpt_dir, "vecnormalize.pkl")
        try:
            self.training_env.save(vec_path)
            if self.verbose:
                print(f"[vecnorm] saved: {vec_path}")
            return vec_path
        except Exception:
            return ""

    def _on_step(self) -> bool:
        """Triggered after each call to env.step() during training."""
        if self.next_idx < len(self.checkpoints) and self.num_timesteps >= self.checkpoints[self.next_idx]:
            t = self.checkpoints[self.next_idx]

            model_path = os.path.join(self.ckpt_dir, f"ppo_{ENV_ID}_{t}.zip")
            self.model.save(model_path)
            print(f"[ckpt] saved: {model_path}")

            vec_path = self._save_vecnormalize()

            video_path = os.path.join(self.video_dir, video_filename(ENV_ID, t))
            record_video(self.model, vec_path, video_path)

            self.next_idx += 1

        return True


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ENV_ID 폴더를 먼저 만들고, 그 안에 날짜(run_id) 폴더 생성
    env_tag = ENV_ID.lower().replace("-", "_")
    env_dir = os.path.join("logs", env_tag)
    os.makedirs(env_dir, exist_ok=True)

    run_dir = os.path.join(env_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[run] directory: {run_dir}")

    seed = 42
    n_envs = max(1, min(8, cpu_count() - 1))
    n_steps = 2048
    batch_size = 2048
    total_timesteps = 10_000_000

    env = SubprocVecEnv([make_env(i, seed=seed) for i in range(n_envs)], start_method="spawn")
    env = VecMonitor(env, filename=os.path.join(run_dir, "monitor.csv"))

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        ortho_init=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=2,
        tensorboard_log=os.path.join(run_dir, "tb"),
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    cb = CheckpointAndVideoCallback(
        checkpoints=[total_timesteps//10, total_timesteps//5, total_timesteps//2],
        run_dir=run_dir,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=cb)

    final_model_path = os.path.join(ckpt_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"[final] saved: {final_model_path}")

    vec_path = os.path.join(ckpt_dir, "vecnormalize.pkl")
    env.save(vec_path)
    print(f"[final] vecnormalize saved: {vec_path}")

    env.close()


if __name__ == "__main__":
    main()
