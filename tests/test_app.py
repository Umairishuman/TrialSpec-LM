"""Task 6 app logic without a model: splitting, validation, aggregation rule, retry/fallback, export, config."""
import json

import pytest

from src.app import aggregate as A
from src.app.backend import GenOut, salvage_evidence_ids
from src.app.config import resolve_model
from src.app.examples import check_no_test, load_examples
from src.app.examples import test_patients as split_test_patients
from src.app.pipeline import Criterion, prepare, run_criterion, screen
from src.app.render import messages_html, results_html
from src.app.report import report_html, report_markdown, write_reports
from src.app.text import parse_criteria, split_note
from src.app.validate import validate_inputs, validate_prompt_lengths

LIMITS = {"max_note_chars": 6000, "max_sentences": 60, "max_criteria": 20, "max_criterion_chars": 600,
          "max_prompt_tokens": 1024, "warn_prompt_tokens": 640}
DECODING = {"max_new_tokens": 64, "retry_max_new_tokens": 128, "repetition_penalty": 1.0}
CFG = {"decoding": DECODING, "limits": LIMITS, "config_path": "configs/app.yaml"}
SENTS = [(0, "A 60-year-old woman with diabetes."), (1, "She had a stroke in 2019."), (2, "No allergies.")]


class ScriptedBackend:
    """Stands in for ModelBackend: returns scripted generations, records calls."""
    info = {"model_id": "scripted", "adapter": "none", "adapter_source": "test", "prompt_version": "v1",
            "draft_model": "none", "dtype": "float32", "device": "cpu", "load_seconds": 0.0}

    def __init__(self, outputs=(), scores=None, tokens=100):
        self.outputs, self.scores, self.tokens = list(outputs), scores, tokens
        self.generate_calls, self.score_calls = [], []

    def prompt(self, sentences, ctype, text):
        return f"{ctype}|{text}|{len(sentences)}"

    def count_tokens(self, prompt):
        return self.tokens

    def generate(self, prompt, max_new_tokens, repetition_penalty=1.0):
        self.generate_calls.append(max_new_tokens)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def score_labels(self, prompt, ctype, evidence_ids):
        self.score_calls.append(list(evidence_ids))
        return self.scores or {"excluded": -9.0, "not excluded": -1.0, "not enough information": -3.0, "not applicable": -5.0}


def ok(text):
    return GenOut(text, 12, 0.1, True)


# ----------------------------------------------------------------------------- sentence splitting
def test_numbered_note_keeps_original_ids():
    sents, mode = split_note("0. First sentence.\n1. Second one.\n5. Gap in IDs is kept.")
    assert mode == "numbered" and [i for i, _ in sents] == [0, 1, 5]
    assert sents[2][1] == "Gap in IDs is kept."


def test_numbered_with_duplicate_ids_is_split_instead():
    sents, mode = split_note("1. One.\n1. Again one.")
    assert mode == "split" and [i for i, _ in sents] == [0, 1]


def test_free_text_split_respects_abbreviations_decimals_initials():
    note = ("A 45 y.o. male, seen by Dr. Lee on 3.5 mg of drug X b.i.d. today. He reports pain, e.g. at night. "
            "J. Smith confirmed it! Is it chronic? Yes.")
    sents, mode = split_note(note)
    assert mode == "split"
    assert [s for _, s in sents] == [
        "A 45 y.o. male, seen by Dr. Lee on 3.5 mg of drug X b.i.d. today.",
        "He reports pain, e.g. at night.", "J. Smith confirmed it!", "Is it chronic?", "Yes."]
    assert [i for i, _ in sents] == [0, 1, 2, 3, 4]


def test_free_text_joins_wrapped_lines_and_breaks_on_bullets():
    note = "The patient has a long history\nof asthma since childhood.\n- Smoker\n- No diabetes\n\nLast line"
    assert [s for _, s in split_note(note)[0]] == [
        "The patient has a long history of asthma since childhood.", "Smoker", "No diabetes", "Last line"]


def test_empty_note():
    assert split_note("  \n ") == ([], "empty")


def test_parse_criteria_strips_markers_drops_blanks_and_duplicates():
    items, notices = parse_criteria("1. Age >= 18\n\n- Diabetes\n(a) HbA1c 7-10%\nage >= 18\nE. coli infection\n  ")
    assert items == ["Age >= 18", "Diabetes", "HbA1c 7-10%", "E. coli infection"]
    assert len(notices) == 1 and "Duplicate" in notices[0]


# ----------------------------------------------------------------------------- validation
@pytest.mark.parametrize("note,inc,exc,needle", [
    ("", ["a"], [], "note is empty"),
    ("x " * 4000, ["a"], [], "too long"),
    ("A note.", [], [], "No criteria"),
    ("A note.", ["c"] * 21, [], "limit is 20"),
    ("A note.", ["x" * 601], [], "Inclusion criterion 1 is too long"),
])
def test_validation_errors(note, inc, exc, needle):
    sents, _ = split_note(note)
    c = validate_inputs(note, sents, inc, exc, LIMITS)
    assert not c.ok and any(needle in e for e in c.errors)


def test_validation_too_many_sentences():
    note = " ".join(f"Sentence {i}." for i in range(61))
    c = validate_inputs(note, split_note(note)[0], ["a"], [], LIMITS)
    assert any("61 sentences" in e for e in c.errors)


def test_validation_ok_and_no_inclusion_warning():
    c = validate_inputs("A note.", [(0, "A note.")], [], ["x"], LIMITS)
    assert c.ok and any("No inclusion" in w for w in c.warnings)


def test_prompt_length_error_and_warning():
    c = validate_prompt_lengths({"I1": 1100, "E1": 700, "E2": 300}, LIMITS)
    assert not c.ok and "I1" in c.errors[0] and "1100" in c.errors[0]
    assert "E1" in c.warnings[0] and "E2" not in c.warnings[0]


def test_prepare_blocks_long_prompt_before_generation():
    b = ScriptedBackend(tokens=2000)
    prep = prepare("A note.", "crit", "", b, LIMITS)
    assert not prep.check.ok and b.generate_calls == []


# ----------------------------------------------------------------------------- aggregation rule (D-029)
def R(cid, ctype, label, ev=(1,), fallback=False):
    return {"id": cid, "criterion_type": ctype, "label": label, "evidence_ids": list(ev), "fallback": fallback}


@pytest.mark.parametrize("results,verdict", [
    ([R("I1", "inclusion", "included"), R("E1", "exclusion", "not excluded")], A.LIKELY_ELIGIBLE),
    ([R("I1", "inclusion", "included"), R("I2", "inclusion", "not applicable")], A.LIKELY_ELIGIBLE),     # N/A neutral
    ([R("I1", "inclusion", "not included"), R("E1", "exclusion", "not enough information")], A.NOT_ELIGIBLE),  # failure wins
    ([R("I1", "inclusion", "included"), R("E1", "exclusion", "excluded")], A.NOT_ELIGIBLE),
    ([R("I1", "inclusion", "included"), R("E1", "exclusion", "not enough information")], A.NEEDS_REVIEW),
    ([R("I1", "inclusion", "included"), R("E1", "exclusion", "excluded", fallback=True)], A.NEEDS_REVIEW),  # fallback = unresolved
    ([R("I1", "inclusion", None)], A.NEEDS_REVIEW),
    ([R("E1", "exclusion", "not excluded")], A.NEEDS_REVIEW),                                              # no inclusion met
    ([R("I1", "inclusion", "not applicable"), R("E1", "exclusion", "not applicable")], A.NEEDS_REVIEW),
])
def test_aggregation_rule(results, verdict):
    assert A.aggregate(results).verdict == verdict


def test_aggregation_lists_unresolved_with_blocking_and_flags_missing_and_no_evidence():
    o = A.aggregate([R("I1", "inclusion", "not included"), R("I2", "inclusion", "not enough information", ev=()),
                     R("E1", "exclusion", "not excluded", ev=())])
    assert o.blocking == ["I1"] and o.unresolved == ["I2"]
    assert o.missing_information == ["I2"] and o.no_evidence == ["E1"]
    assert any("Also unresolved: I2" in r for r in o.reasons)
    assert "not ground truth" in o.rule


def test_aggregation_empty_raises():
    with pytest.raises(ValueError):
        A.aggregate([])


# ----------------------------------------------------------------------------- retry / fallback (D-028)
EXC = Criterion("E1", "exclusion", "History of stroke")


def test_valid_json_no_retry_no_fallback():
    b = ScriptedBackend([ok('{"evidence_ids": [1], "label": "excluded"}')])
    r = run_criterion(b, SENTS, EXC, "p", DECODING)
    assert (r["label"], r["status"], r["evidence_ids"]) == ("excluded", "ok", [1])
    assert b.generate_calls == [64] and b.score_calls == []
    assert r["evidence"][0]["text"] == "She had a stroke in 2019." and "[1]" in r["explanation"]


def test_truncated_output_is_retried_with_larger_budget():
    b = ScriptedBackend([GenOut('{"evidence_ids": [1, 2', 64, 1.0, False), ok('{"evidence_ids": [1], "label": "excluded"}')])
    r = run_criterion(b, SENTS, EXC, "p", DECODING)
    assert r["status"] == "retried" and r["label"] == "excluded" and b.generate_calls == [64, 128]
    assert r["flags"]["retried"] and not r["fallback"]


def test_malformed_output_falls_back_to_label_scoring_with_salvaged_ids():
    b = ScriptedBackend([ok('{"evidence_ids": [1, 9], "label": "maybe"}')])   # ended at EOS: no retry
    r = run_criterion(b, SENTS, EXC, "p", DECODING)
    assert b.generate_calls == [64] and b.score_calls == [[1]]
    assert r["status"] == "fallback" and r["label"] == "not excluded" and r["fallback"]
    assert r["evidence_ids"] == [1] and r["invalid_ids"] == [9] and r["flags"]["hallucinated_ids"]
    assert r["aggregate_status"] == A.UNRESOLVED and "fallback" in r["explanation"]


def test_model_exception_becomes_unresolved_row_and_screening_continues():
    b = ScriptedBackend([RuntimeError("out of memory"), ok('{"evidence_ids": [0], "label": "included"}')])
    prep = prepare("0. A 60-year-old woman with diabetes.\n1. She had a stroke in 2019.", "Diabetes", "Stroke", b, LIMITS)
    # inclusion I1 runs first and fails; exclusion E1 still runs
    s = screen(prep, b, CFG)
    i1, e1 = s.results
    assert i1["status"] == "error" and i1["label"] is None and "out of memory" in i1["explanation"]
    assert e1["status"] == "fallback"          # "included" is not an exclusion label -> fallback
    assert s.outcome.verdict == A.NEEDS_REVIEW


def test_no_evidence_flag_for_definite_label():
    b = ScriptedBackend([ok('{"evidence_ids": [], "label": "not excluded"}')])
    r = run_criterion(b, SENTS, EXC, "p", DECODING)
    assert r["flags"]["no_evidence"] and "No sentence was cited" in r["explanation"]


@pytest.mark.parametrize("text,ids", [
    ('{"evidence_ids": [3, 4], "label": "inc', [3, 4]),
    ('"evidence_ids": [ 2 ,5 ]', [2, 5]),
    ('{"evidence_ids": ["1", 2]}', [2]),
    ("garbage", []),
])
def test_salvage_evidence_ids(text, ids):
    assert salvage_evidence_ids(text) == ids


# ----------------------------------------------------------------------------- export (G4)
def _screening():
    b = ScriptedBackend([ok('{"evidence_ids": [0], "label": "included"}'),
                         ok('{"evidence_ids": [], "label": "not excluded"}')])
    note = "0. A 60-year-old woman <script>alert(1)</script> | pipe.\n1. No stroke."
    prep = prepare(note, "Age > 50", "Stroke", b, LIMITS)
    return screen(prep, b, CFG)


def test_report_markdown_has_disclaimer_outcome_and_provenance():
    md = report_markdown(_screening())
    assert "Educational screening prototype — not a clinical enrollment decision." in md
    assert "Likely eligible (preliminary)" in md and "rule-based aggregation" in md
    for field in ("scripted", "Prompt version", "repetition_penalty", "Created (UTC)", "Git commit"):
        assert field in md
    assert "\\|" in md     # pipes in the note are escaped inside tables/lists


def test_report_html_escapes_input_and_is_standalone(tmp_path):
    s = _screening()
    html = report_html(s, gold={("inclusion", "Age > 50"): "included"})
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert html.startswith("<!doctype html>") and "Expert label" in html and "✓" in html
    md, h = write_reports(s, tmp_path)
    assert md.exists() and h.exists() and md.suffix == ".md" and h.suffix == ".html"


def test_render_escapes_messages_and_results():
    assert "&lt;b&gt;" in messages_html(["<b>x</b>"])
    assert "<script>" not in results_html(_screening().results)


# ----------------------------------------------------------------------------- config + examples
def test_resolve_model_defaults_to_app_yaml(tmp_path):
    cfg = {"use_final_selection": True, "final_selection": str(tmp_path / "missing.json"),
           "adapter": "a/run1", "prompt_version": "v1"}
    r = resolve_model(cfg)
    assert (r["adapter"], r["prompt_version"]) == ("a/run1", "v1") and "app.yaml" in r["source"]


def test_resolve_model_uses_final_selection(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"selected": "run4", "adapter": "a/run4",
                                                         "created_utc": "2026-11-10T00:00:00+00:00"}))
    (tmp_path / "fe.yaml").write_text("candidates:\n  - {name: run1, adapter: a/run1, prompt_version: v1}\n"
                                      "  - {name: run4, adapter: a/run4, prompt_version: v2}\n")
    cfg = {"use_final_selection": True, "final_selection": str(tmp_path / "selection.json"),
           "final_eval_config": str(tmp_path / "fe.yaml"), "adapter": "a/run1", "prompt_version": "v1"}
    r = resolve_model(cfg)
    assert (r["adapter"], r["prompt_version"], r["run"]) == ("a/run4", "v2", "run4")
    cfg["use_final_selection"] = False
    assert resolve_model(cfg)["adapter"] == "a/run1"


def test_examples_never_use_test_patients():
    examples = load_examples()
    assert examples, "data/app_examples.json missing (python -m src.app.examples)"
    assert {e["split"] for e in examples} <= {"train", "val"}
    assert not ({e["patient_id"] for e in examples} & split_test_patients())
    with pytest.raises(ValueError):
        check_no_test([{"patient_id": next(iter(split_test_patients()))}])
