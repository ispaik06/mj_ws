# {{PROJECT_NAME_RAW}}

A project template for training and evaluating a custom MuJoCo environment with **Gymnasium** + **Stable-Baselines3 (PPO)**.
The environment `ENV_ID={{ENV_ID}}` is registered automatically via the `envs` package, and you can run training/playback immediately using `scripts/train.py` and `scripts/play.py`.

---

## Project Structure

```
.
├─ source/
│  ├─ envs/
│  │  ├─ __init__.py         # Registers the Gymnasium env (register)
│  │  └─ {{ENV_MODULE}}.py   # Environment class ({{ENV_CLASS}})
│  ├─ assets/
│  │  └─ humanoid.xml        # MuJoCo XML assets (optional / recommended)
│  └─ __init__.py
├─ scripts/
│  ├─ train.py               # PPO training entrypoint
│  └─ play.py                # Load latest run and render
├─ logs/                     # Training outputs (auto-created)
└─ pyproject.toml
```

---

## Requirements

* Python 3.10+ recommended
* MuJoCo (via `pip install mujoco`)
* Gymnasium
* Stable-Baselines3

> Tip: Use a virtual environment (Conda/venv) to avoid dependency conflicts.

---

## Setup

From the project root:

```bash
pip install -e .
```

This installs the project in **editable mode**, so any changes you make in `source/` are picked up immediately (no reinstall needed).

---

## Train

```bash
python scripts/train.py
```

Training outputs are written to:

```
logs/<env_tag>/<YYYYMMDD_HHMMSS>/
  ├─ checkpoints/
  ├─ videos/
  ├─ monitor.csv
  └─ tb/                  # TensorBoard logs
```

To view TensorBoard:

```bash
tensorboard --logdir logs
```

---

## Play (Render the Latest Run)

```bash
python scripts/play.py
```

This script automatically finds the most recent run under `logs/<env_tag>/...`, loads:

* `checkpoints/final_model.zip`
* `checkpoints/vecnormalize.pkl`

and renders the policy with `render_mode="human"`.

---

## Customizing the Environment

### 1) Environment registration

Environment registration happens in:

* `source/envs/__init__.py`

The `ENV_ID={{ENV_ID}}` is registered with:

* `entry_point="envs.{{ENV_MODULE}}:{{ENV_CLASS}}"`

### 2) Environment implementation

Your environment class lives in:

* `source/envs/{{ENV_MODULE}}.py`

If you use a custom MuJoCo XML, store it under:

* `source/assets/`

and build a robust path in your env (recommended):

* resolve from `__file__` rather than relying on the current working directory.

---

## Notes

* If you use `SubprocVecEnv(start_method="spawn")`, environment registration must happen in **each worker process**.
  This template ensures that by importing `envs` inside `make_env()`.

* If you delete the project folder after running `pip install -e .`, you may end up with a “broken editable install”.
  Clean up with:

  ```bash
  pip uninstall {{PROJECT_SLUG}}
  ```

---

## Quick Commands

```bash
pip install -e .
python scripts/train.py
python scripts/play.py
tensorboard --logdir logs
```
