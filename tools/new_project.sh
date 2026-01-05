#!/usr/bin/env bash
set -euo pipefail

RAW_NAME="${1:-}"
TEMPLATE_NAME="${2:-mujoco_sb3_base}"

if [[ -z "${RAW_NAME}" ]]; then
  echo 'Usage: ./tools/new_project.sh "<project name>" [template_name]'
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${ROOT_DIR}/templates/${TEMPLATE_NAME}"

if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "[ERR] Template not found: ${TEMPLATE_DIR}"
  exit 1
fi

# ✅ IMPORTANT: pass shell variables to the Python process
export ROOT_DIR
export TEMPLATE_DIR
export RAW_NAME

python - <<'PY'
import os, re, pathlib, shutil, sys

root = os.environ["ROOT_DIR"]
template_dir = os.environ["TEMPLATE_DIR"]
raw = os.environ["RAW_NAME"]

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)         # spaces/hyphens -> _
    s = re.sub(r"[^0-9a-z_]+", "_", s)     # other chars -> _
    s = re.sub(r"_+", "_", s).strip("_")   # collapse underscores
    if not s:
        raise ValueError("Project name becomes empty after normalization.")
    return s

def pascal_from_slug(slug: str) -> str:
    parts = [p for p in slug.split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)

slug = slugify(raw)
env_id = f"{slug}-v0"
env_class = f"{pascal_from_slug(slug)}Env"
env_module = slug

projects_dir = pathlib.Path(root) / "projects"
out_dir = projects_dir / slug

if out_dir.exists():
    print(f"[ERR] Project already exists: {out_dir}")
    sys.exit(1)

projects_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(template_dir, out_dir)

# 1) rename env file: env_template.py -> <slug>.py
env_template = out_dir / "source" / "envs" / "env_template.py"
env_target = out_dir / "source" / "envs" / f"{env_module}.py"
if env_template.exists():
    env_template.rename(env_target)

# 2) replace placeholders in text files
repls = {
    "{{PROJECT_NAME_RAW}}": raw,
    "{{PROJECT_SLUG}}": slug,
    "{{ENV_ID}}": env_id,
    "{{ENV_CLASS}}": env_class,
    "{{ENV_MODULE}}": env_module,
}

for p in out_dir.rglob("*"):
    if p.is_file():
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for k, v in repls.items():
            txt = txt.replace(k, v)
        p.write_text(txt, encoding="utf-8")

print(f"[OK] Created project: {out_dir}")
print(f"     project_slug : {slug}")
print(f"     env_id       : {env_id}")
print(f"     env_module   : envs.{env_module}")
print(f"     env_class    : {env_class}")
print("")
print("Next steps:")
print(f"  cd {out_dir}")
print("  pip install -e .")
print("  python scripts/train.py")
PY
