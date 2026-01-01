import os
from datetime import datetime

import gymnasium as gym
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


ENV_ID = "Humanoid-v5"


def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)  # episode reward/len 기록용
        env.reset(seed=seed + rank)
        return env
    return _init


def record_video(model, vecnorm_path: str, out_mp4: str, max_steps: int = 2000, fps: int = 30):
    # render 가능한 base env
    base_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

    # VecNormalize 로드(학습 통계 적용)
    if vecnorm_path and os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, base_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = base_env  # fallback

    obs = eval_env.reset()

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
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
    def __init__(self, checkpoints, run_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoints = sorted(checkpoints)
        self.run_dir = run_dir
        self.next_idx = 0

        # ✅ 모든 산출물을 run_dir/checkpoints 아래로
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.video_dir = os.path.join(run_dir, "videos")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def _save_vecnormalize(self) -> str:
        # ✅ vecnormalize도 checkpoints 폴더 안에 저장
        vec_path = os.path.join(self.ckpt_dir, "vecnormalize.pkl")
        try:
            self.training_env.save(vec_path)
            if self.verbose:
                print(f"[vecnorm] saved: {vec_path}")
            return vec_path
        except Exception:
            return ""

    def _on_step(self) -> bool:
        if self.next_idx < len(self.checkpoints) and self.num_timesteps >= self.checkpoints[self.next_idx]:
            t = self.checkpoints[self.next_idx]

            # 1) 모델 저장
            model_path = os.path.join(self.ckpt_dir, f"ppo_{ENV_ID}_{t}.zip")
            self.model.save(model_path)
            print(f"[ckpt] saved: {model_path}")

            # 2) VecNormalize 통계 저장
            vec_path = self._save_vecnormalize()

            # 3) 영상 저장
            video_path = os.path.join(self.video_dir, f"ppo_{ENV_ID}_{t}.mp4")
            record_video(self.model, vec_path, video_path)

            self.next_idx += 1

        return True


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ✅ 최종 저장 폴더를 미리 만들어 둠
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[run] directory: {run_dir}")

    seed = 42
    n_envs = 16
    n_steps = 2048
    batch_size = 2048
    total_timesteps = 10_000_000

    env = SubprocVecEnv([make_env(i, seed=seed) for i in range(n_envs)], start_method="fork")
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
        checkpoints=[500_000, 2_000_000, 5_000_000],
        run_dir=run_dir,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=cb)

    # ✅ final_model.zip도 날짜시간폴더/checkpoints/ 안에 저장
    final_model_path = os.path.join(ckpt_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"[final] saved: {final_model_path}")

    # ✅ vecnormalize.pkl도 checkpoints 폴더 안에 저장
    vec_path = os.path.join(ckpt_dir, "vecnormalize.pkl")
    env.save(vec_path)
    print(f"[final] vecnormalize saved: {vec_path}")

    env.close()


if __name__ == "__main__":
    main()
