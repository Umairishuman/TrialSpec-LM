import json
import sys

import yaml

from src.eval import select_checkpoint


def fake_metrics(macro_f1, acc):
    block = {"accuracy": acc, "macro_f1": macro_f1}
    return {"meta": {}, "metrics": {"overall": block, "inclusion": block, "exclusion": block,
                                    "evidence": {"micro": {"f1": 0.5}}}}


def test_selects_best_val_macro_f1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = "r"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"run_name": run}))
    res = tmp_path / "results/finetuned"
    res.mkdir(parents=True)
    scores = {38: (0.30, 0.70), 76: (0.45, 0.72), 114: (0.45, 0.74)}
    for step, (f1, acc) in scores.items():
        ck = tmp_path / f"outputs/{run}/checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "adapter_config.json").write_text("{}")
        (ck / "adapter_model.safetensors").write_text(str(step))
        (res / f"{run}_checkpoint-{step}_val.metrics.json").write_text(json.dumps(fake_metrics(f1, acc)))
        (res / f"{run}_checkpoint-{step}_val.jsonl").write_text("")
    (tmp_path / f"outputs/{run}/final").mkdir()
    (tmp_path / f"outputs/{run}/final/tokenizer.json").write_text("{}")

    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml"])
    select_checkpoint.main()

    sel = json.loads((res / f"{run}_selection.json").read_text())
    assert sel["selected"]["checkpoint"] == "checkpoint-114"        # tie on F1 -> higher accuracy
    assert (tmp_path / f"outputs/{run}/selected/adapter_model.safetensors").read_text() == "114"
    assert (tmp_path / f"outputs/{run}/selected/tokenizer.json").exists()
    assert json.loads((res / f"{run}_val.metrics.json").read_text())["metrics"]["overall"]["accuracy"] == 0.74
