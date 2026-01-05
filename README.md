# mj_ws

A lightweight MuJoCo RL workspace inspired by the IsaacLab workflow.

* Keep **shared tooling and templates** in this repo (`tools/`, `templates/`).
* Create **project folders** under `projects/` using a single command.
* Each project is a self-contained package (editable install) with:

  * `source/envs/` (Gymnasium env registration + implementation)
  * `scripts/train.py` / `scripts/play.py`
  * `logs/` output directory (checkpoints, videos, TensorBoard)

---

## Workspace Layout

```
mj_ws/
├─ templates/
│  └─ mujoco_sb3_base/             # project template
│     ├─ source/
│     │  ├─ envs/
│     │  │  ├─ __init__.py         # env registration template
│     │  │  └─ env_template.py     # env implementation template
│     │  ├─ assets/
│     │  │  └─ humanoid.xml        # example MuJoCo XML (optional)
│     │  └─ __init__.py
│     ├─ scripts/
│     │  ├─ train.py               # PPO training template
│     │  └─ play.py                # playback template
│     └─ README.md                 # per-project README template
├─ tools/
│  └─ new_project.sh               # create a new project from a template
├─ projects/                       # generated projects live here
└─ README.md
```

---

## Prerequisites / Installation

> Recommended: create a dedicated Conda environment for this workspace.

### 0) Create & activate environment

```bash
conda create -n mujoco python=3.11 -y
conda activate mujoco
```

### 1) Install system/graphics dependencies (Conda)

```bash
conda install -c conda-forge glfw -y
conda install -c conda-forge pyopengl -y
```

### 2) Install Python packages

```bash
pip install -U pip

pip install mujoco gymnasium
python -m pip install stable-baselines3 "gymnasium[mujoco]"

pip install tensorboard
```

### 3) Install PyTorch (CPU build)

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> If you want GPU acceleration, install a CUDA-enabled PyTorch build that matches your OS/CUDA setup instead of the CPU build above.

---

## Quick Start

### 1) Create a new project

From the **workspace root**:

```bash
./tools/new_project.sh MyHumanoid
```

This creates:

```
projects/myhumanoid/
```

The generator auto-synchronizes:

* env file name (`source/envs/myhumanoid.py`)
* env class name (e.g., `MyHumanoidEnv`)
* env id (e.g., `myhumanoid-v0`)
* `scripts/train.py` / `scripts/play.py` use the same env id

### 2) Install the project (editable)

```bash
cd projects/myhumanoid
pip install -e .
```

### 3) Train

```bash
python scripts/train.py
```

### 4) Play (render the latest run)

```bash
python scripts/play.py
```

---

## How Projects Work

Each generated project is a Python package installed in **editable mode** (`pip install -e .`).

* ✅ You can `import envs` and `gym.make(ENV_ID)` from anywhere
* ✅ Changes in `source/` are reflected immediately (no reinstall)
* ⚠️ If you delete the project folder without uninstalling, you may leave a “broken editable install”

To remove a project cleanly:

```bash
pip uninstall <project_package_name>
rm -rf projects/<project_slug>
```

---

## Where Logs Go

Training writes outputs to:

```
projects/<project_slug>/logs/<env_tag>/<YYYYMMDD_HHMMSS>/
  ├─ checkpoints/
  ├─ videos/
  ├─ monitor.csv
  └─ tb/
```

TensorBoard:

```bash
tensorboard --logdir projects/<project_slug>/logs
```

---

## Custom MuJoCo XML Assets

Store per-project XML and related files under:

```
projects/<project_slug>/source/assets/
```

Recommended: resolve XML paths **relative to the env file** (via `__file__`) rather than relying on the current working directory.

---

## Git / Version Control Notes

Typical `.gitignore` entries for this workspace:

* `projects/**/logs/`
* `projects/**/__pycache__/`
* `projects/**/.venv/`
* `**/*.zip` (model checkpoints)
* `**/*.pkl` (VecNormalize stats)
* `**/*.mp4` (videos)
* `.DS_Store`

Recommended workflow:

* Keep this repo (`mj_ws`) focused on **tools/templates**.
* Treat each generated project as either:

  * its own repo, or
  * a local experiment folder ignored by git.

---

## Troubleshooting

### “Environment <id> doesn’t exist”

* Ensure `import envs` is executed before `gym.make(ENV_ID)`.
* With `SubprocVecEnv(start_method="spawn")`, env registration must happen in every worker process.
  This workspace template imports `envs` inside the worker initializer for that reason.

### “Broken editable install”

If you removed a project folder but it still appears installed:

```bash
pip uninstall <project_package_name>
```

---
