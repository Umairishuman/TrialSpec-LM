"""Final-evaluation helpers (D-024): selection rule, test lock, explanations, error categories, plots."""
import json
from pathlib import Path

import pytest

from src.eval.error_analysis import categories, render, sample_cases
from src.eval.explain import NO_EVIDENCE, UNPARSEABLE, evidence_text, explanation
from src.eval.final_eval import check_test_lock, select_best
from src.eval.metrics import full_report


def res(f1, acc):
    return {"meta": {}, "metrics": {"overall": {"macro_f1": f1, "accuracy": acc}}}


def test_select_best_rule():
    order = ["run1", "run2", "run3", "run4"]
    assert select_best({"run1": res(0.40, 0.7), "run2": res(0.45, 0.6), "run3": res(0.30, 0.9), "run4": res(0.44, 0.9)}, order) == "run2"
    # tie on F1 -> accuracy
    assert select_best({"run1": res(0.4, 0.7), "run2": res(0.4, 0.8), "run3": res(0.1, 0.1), "run4": res(0.1, 0.1)}, order) == "run2"
    # full tie -> earlier in config order
    assert select_best({n: res(0.4, 0.7) for n in order}, order) == "run1"


def test_test_lock_blocks_second_run(tmp_path):
    check_test_lock(tmp_path, allow_rerun=False)              # no lock yet: fine
    (tmp_path / "test_lock.json").write_text(json.dumps({"created_utc": "x"}))
    with pytest.raises(SystemExit):
        check_test_lock(tmp_path, allow_rerun=False)
    check_test_lock(tmp_path, allow_rerun=True)


def test_explanation_uses_cited_sentences_only():
    sents = [(0, "65 year old man."), (1, "History of angina."), (2, "No diabetes.")]
    cited = evidence_text(sents, [1, 7])                      # 7 is not in the note -> skipped
    assert cited == [{"id": 1, "text": "History of angina."}]
    e = explanation("inclusion", "included", cited)
    assert "[1]" in e and "History of angina." in e and "65 year" not in e
    assert explanation("exclusion", "not excluded", []).endswith(NO_EVIDENCE)
    assert explanation("inclusion", None, []) == UNPARSEABLE


def rec(i, ctype, gold, pred, ge=(), pe=(), invalid=(), ok=True):
    return {"annotation_id": i, "patient_id": "p", "trial_id": "t", "criterion_type": ctype, "criterion_text": "c",
            "gold_label": gold, "pred_label": pred, "gold_evidence": list(ge), "pred_evidence": list(pe),
            "invalid_ids": list(invalid), "parse_ok": ok, "raw": "{}"}


RECORDS = [
    rec(0, "inclusion", "included", "not included", [1], [1]),                    # polarity flip
    rec(1, "exclusion", "not enough information", "not excluded"),                # missed information
    rec(2, "inclusion", "included", "not enough information", [2]),               # spurious NEI
    rec(3, "exclusion", "not applicable", "not excluded"),                        # applicability
    rec(4, "inclusion", "included", "included", [1], [2]),                        # unsupported evidence
    rec(5, "exclusion", "not excluded", "not excluded", [], [0], invalid=[9]),    # unsupported + hallucinated
    rec(6, "inclusion", "included", None, ok=False),                              # parse failure
    rec(7, "inclusion", "included", "included", [1, 2], [2]),                     # correct
    rec(8, "exclusion", "excluded", "not excluded", [0]),                         # polarity flip #2
]


def test_error_categories():
    cats = [categories(r) for r in RECORDS]
    assert cats[0] == ["polarity_flip"]
    assert cats[1] == ["missed_information"]
    assert cats[2] == ["spurious_nei"]
    assert cats[3] == ["applicability"]
    assert cats[4] == ["unsupported_evidence"]
    assert cats[5] == ["unsupported_evidence", "hallucinated_ids"]
    assert cats[6] == ["parse_failure"]
    assert cats[7] == []


def test_sampling_is_distinct_and_spread():
    picked = sample_cases(RECORDS, 8)
    ids = [r["annotation_id"] for _, r in picked]
    assert len(ids) == len(set(ids)) == 8 and 7 not in ids                       # never the correct record
    assert {c for c, _ in picked[:7]} >= {"polarity_flip", "missed_information", "spurious_nei",
                                          "applicability", "unsupported_evidence", "parse_failure"}
    md = render(RECORDS, {0: [(0, "a"), (1, "b")]}, "t", 8)
    assert md.count("## Case ") == 8 and "[GP] 1. b" in md


def test_metrics_count_invalid_ids():
    e = full_report(RECORDS)["evidence"]
    assert e["n_invalid_ids"] == 1 and e["n_records_with_invalid_ids"] == 1


def test_orchestrator_end_to_end_with_fake_model(tmp_path, monkeypatch):
    """Runs final_eval.main() with a fake model: checks selection, test-once lock and every artifact."""
    pytest.importorskip("matplotlib")
    import sys

    import pandas as pd
    import yaml

    from src.eval import final_eval

    sents = [(0, "a"), (1, "b")]
    def frame(n):
        return pd.DataFrame([{"annotation_id": i, "sentences": sents} for i in range(n)])
    monkeypatch.setattr(final_eval, "load_split_frames", lambda: {"val": frame(6), "test": frame(5)})
    monkeypatch.setattr(final_eval, "load_model", lambda *a, **k: (None, object()))
    quality = {"zeroshot-v1": 0, "zeroshot-v2": 0, "run1": 3, "run2": 5, "run4": 5}   # run2/run4 tie on val

    def fake_eval(tok, model, fr, out, *, adapter, split, extra_meta, **kw):
        name = extra_meta["final_eval_name"]
        recs = [rec(int(i), "inclusion", "included", "included" if i < quality[name] else "not included", [1], [1])
                for i in fr["annotation_id"]]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        result = {"meta": {"prompt_version": "v1", "parse_ok_rate": 1.0, "git_commit": "abcdef0", "split": split},
                  "metrics": full_report(recs)}
        Path(f"{out}.metrics.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(final_eval, "evaluate_and_write", fake_eval)
    monkeypatch.setattr(final_eval, "write_error_analysis",
                        lambda pred, split, out, n: out.write_text(render(
                            [json.loads(l) for l in pred.read_text().splitlines()], {}, "t", n)))

    monkeypatch.chdir(tmp_path)
    cfg = {"model_id": "m", "out_dir": "results/final", "error_cases": 8,
           "decoding": {"repetition_penalty": 1.0, "max_new_tokens": 64, "batch_size": 8},
           "baselines": [{"name": "zeroshot-v1", "adapter": None, "prompt_version": "v1"},
                         {"name": "zeroshot-v2", "adapter": None, "prompt_version": "v2"}],
           "candidates": [{"name": "run1", "adapter": "a1", "prompt_version": "v1"},
                          {"name": "run2", "adapter": "a2", "prompt_version": "v2"},
                          {"name": "run4", "adapter": "a4", "prompt_version": "v1"}]}
    Path("c.yaml").write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml", "--smoke"])
    final_eval.main()
    assert not list(Path("results/smoke/final").glob("*_test*"))              # smoke never touches test
    assert not Path("results/final/test_lock.json").exists()

    monkeypatch.setattr(sys, "argv", ["x", "--config", "c.yaml"])
    final_eval.main()
    out = Path("results/final")
    sel = json.loads((out / "selection.json").read_text())
    assert sel["selected"] == "run2" and sel["baseline"] == "zeroshot-v2"    # tie with run4 -> config order
    assert sorted(p.name for p in out.glob("*_test.metrics.json")) == ["run2_test.metrics.json", "zeroshot-v2_test.metrics.json"]
    for f in ["test_lock.json", "summary.md", "error_analysis_val.md", "error_analysis_test.md",
              "run2_test_confusion.png", "run2_test.csv", "run1_val.csv"]:
        assert (out / f).exists(), f
    assert "## Test: labels" in (out / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):                                           # second full run is refused
        final_eval.main()


def test_confusion_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from src.eval.plots import confusion_figure
    f = tmp_path / "m.metrics.json"
    f.write_text(json.dumps({"meta": {}, "metrics": full_report(RECORDS)}))
    out = confusion_figure(f, tmp_path / "m.png")
    assert out.exists() and out.stat().st_size > 1000
