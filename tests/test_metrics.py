import pytest

from src.data.prompt import build_target, parse_output
from src.eval.metrics import classification_report, evidence_scores


def test_evidence_empty_rules():
    s = evidence_scores([[], [], [1]], [[], [2], []])
    assert s["macro"]["f1"] == pytest.approx(1 / 3)   # only the both-empty record scores 1
    assert s["n_both_empty"] == 1 and s["n_gold_empty"] == 2


def test_evidence_micro_vs_macro():
    s = evidence_scores([[1, 2], [3]], [[1], [3, 4]])
    assert s["micro"]["precision"] == pytest.approx(2 / 3)
    assert s["micro"]["recall"] == pytest.approx(2 / 3)
    assert s["macro"]["precision"] == pytest.approx((1 + 0.5) / 2)
    assert s["macro"]["recall"] == pytest.approx((0.5 + 1) / 2)


def test_classification_handles_invalid_and_absent_class():
    labels = ["excluded", "not excluded", "not enough information", "not applicable"]
    r = classification_report(["not excluded", "not excluded", "not applicable"], ["not excluded", None, "not applicable"], labels)
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["invalid_rate"] == pytest.approx(1 / 3)
    assert r["per_class"]["excluded"]["support"] == 0 and "note" in r["per_class"]["excluded"]
    assert r["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)   # macro over present classes only


def test_parse_roundtrip_and_failures():
    valid = {0, 1, 2}
    p = parse_output(build_target("not enough information", [2, 0]), "inclusion", valid)
    assert p["parse_ok"] and p["label"] == "not enough information" and p["evidence_ids"] == [0, 2]
    p = parse_output('{"evidence_ids": [1, 9], "label": "included"}', "inclusion", valid)
    assert p["evidence_ids"] == [1] and p["invalid_ids"] == [9]
    assert not parse_output('{"label": "excluded"}', "inclusion", valid)["parse_ok"]   # wrong label set
    assert not parse_output("I think yes", "inclusion", valid)["parse_ok"]
