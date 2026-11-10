"""App configuration: configs/app.yaml plus the final model selection when it exists (D-028)."""
import json
from pathlib import Path

from src.utils import load_config

DEFAULT_CONFIG = Path("configs/app.yaml")


def resolve_model(cfg: dict) -> dict:
    """Which adapter and prompt version the app uses, and where that choice came from.

    If `use_final_selection` is on and the final evaluation has written its selection file, that selection
    wins (its prompt version comes from the final-eval config's candidate list). Otherwise app.yaml is used.
    """
    sel_path = Path(cfg.get("final_selection", "results/final/selection.json"))
    if cfg.get("use_final_selection") and sel_path.exists():
        sel = json.loads(sel_path.read_text())
        name, adapter = sel["selected"], sel["adapter"]
        fe = load_config(cfg["final_eval_config"])
        versions = {c["name"]: c["prompt_version"] for c in fe.get("candidates", [])}
        if name not in versions:
            raise ValueError(f"{sel_path} selects {name!r}, which is not a candidate in {cfg['final_eval_config']}")
        return {"adapter": adapter, "prompt_version": versions[name], "run": name,
                "source": f"{sel_path.as_posix()} (selected {name}, {sel.get('created_utc', 'unknown time')})"}
    return {"adapter": cfg.get("adapter"), "prompt_version": cfg["prompt_version"], "run": None,
            "source": "configs/app.yaml (default; no final selection found)"}


def load_app_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    cfg = load_config(path)
    cfg["config_path"] = Path(path).as_posix()
    cfg["resolved"] = resolve_model(cfg)
    return cfg
