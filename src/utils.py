import os
import subprocess
from pathlib import Path

import yaml


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def wandb_enabled() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))
