"""Screening workflow: note + criteria -> per-criterion label, evidence, explanation -> aggregated outcome.

Each criterion is one prompt, built and parsed exactly as in training/evaluation (prompt_text, parse_output,
greedy_config). Malformed output: one retry with a larger budget if the answer was cut off, then constrained
label scoring (D-028). Every result keeps the raw model text for inspection.
"""
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.app.aggregate import RULE_TEXT, Outcome, aggregate, criterion_status
from src.app.backend import salvage_evidence_ids
from src.app.text import parse_criteria, split_note
from src.app.validate import Check, validate_inputs, validate_prompt_lengths
from src.data.prompt import parse_output
from src.eval.explain import evidence_text, explanation
from src.utils import git_commit, library_versions

FALLBACK_NOTE = ("The model's answer could not be parsed, so this label was chosen by ranking the four allowed "
                 "answers by model likelihood (fallback); treat it as unresolved.")


@dataclass
class Criterion:
    id: str            # "I1", "E2", ...
    criterion_type: str
    text: str


@dataclass
class Prepared:
    sentences: list[tuple[int, str]]
    split_mode: str
    criteria: list[Criterion]
    prompts: dict[str, str]
    check: Check
    notices: list[str] = field(default_factory=list)


@dataclass
class Screening:
    sentences: list[tuple[int, str]]
    split_mode: str
    results: list[dict]
    outcome: Outcome
    provenance: dict
    warnings: list[str]
    notices: list[str]


def build_criteria(inclusion: list[str], exclusion: list[str]) -> list[Criterion]:
    return ([Criterion(f"I{i}", "inclusion", t) for i, t in enumerate(inclusion, 1)]
            + [Criterion(f"E{i}", "exclusion", t) for i, t in enumerate(exclusion, 1)])


def prepare(note: str, inclusion_text: str, exclusion_text: str, backend, limits: dict) -> Prepared:
    """Split and validate the inputs; build every prompt and check its token length. No generation."""
    sentences, mode = split_note(note)
    inclusion, n1 = parse_criteria(inclusion_text)
    exclusion, n2 = parse_criteria(exclusion_text)
    check = validate_inputs(note, sentences, inclusion, exclusion, limits)
    criteria = build_criteria(inclusion, exclusion)
    prompts = {}
    if check.ok:
        prompts = {c.id: backend.prompt(sentences, c.criterion_type, c.text) for c in criteria}
        check.extend(validate_prompt_lengths({k: backend.count_tokens(p) for k, p in prompts.items()}, limits))
    return Prepared(sentences, mode, criteria, prompts, check, n1 + n2)


def run_criterion(backend, sentences, crit: Criterion, prompt: str, decoding: dict) -> dict:
    valid = {i for i, _ in sentences}
    rp = decoding.get("repetition_penalty", 1.0)
    attempts = []
    status, fallback_scores = "ok", None
    t0 = time.perf_counter()
    try:
        g = backend.generate(prompt, decoding["max_new_tokens"], rp)
        attempts.append({"max_new_tokens": decoding["max_new_tokens"], "raw": g.text, "n_tokens": g.n_tokens,
                         "seconds": round(g.seconds, 2), "hit_eos": g.hit_eos})
        p = parse_output(g.text, crit.criterion_type, valid)
        if not p["parse_ok"] and not g.hit_eos:          # cut off before the JSON closed: one longer retry
            g = backend.generate(prompt, decoding["retry_max_new_tokens"], rp)
            attempts.append({"max_new_tokens": decoding["retry_max_new_tokens"], "raw": g.text, "n_tokens": g.n_tokens,
                             "seconds": round(g.seconds, 2), "hit_eos": g.hit_eos})
            p = parse_output(g.text, crit.criterion_type, valid)
            status = "retried" if p["parse_ok"] else status
        label, ids, invalid = p["label"], p["evidence_ids"], p["invalid_ids"]
        if not p["parse_ok"]:
            salvaged = salvage_evidence_ids(g.text)
            ids = sorted({i for i in salvaged if i in valid})
            invalid = sorted({i for i in salvaged if i not in valid})
            fallback_scores = backend.score_labels(prompt, crit.criterion_type, ids)
            label = max(fallback_scores, key=fallback_scores.get)
            status = "fallback"
        error = None
    except Exception as e:                                # keep going with the other criteria
        traceback.print_exc()
        label, ids, invalid, status, error = None, [], [], "error", f"{type(e).__name__}: {e}"

    cited = evidence_text(sentences, ids)
    expl = explanation(crit.criterion_type, label, cited)
    if status == "fallback":
        expl = f"{FALLBACK_NOTE} {expl}"
    elif status == "error":
        expl = f"The model failed on this criterion ({error}); no decision was made."
    fallback = status in ("fallback", "error")
    return {
        "id": crit.id, "criterion_type": crit.criterion_type, "criterion_text": crit.text,
        "label": label, "evidence_ids": ids, "evidence": cited, "invalid_ids": invalid, "explanation": expl,
        "status": status, "fallback": fallback, "fallback_scores": fallback_scores, "error": error,
        "aggregate_status": criterion_status(crit.criterion_type, label, fallback),
        "flags": {"missing_information": label == "not enough information",
                  "no_evidence": label not in (None, "not enough information") and not fallback and not cited,
                  "hallucinated_ids": bool(invalid), "retried": status == "retried", "fallback": fallback},
        "attempts": attempts, "seconds": round(time.perf_counter() - t0, 2),
    }


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    v = library_versions()
    try:
        v["gradio"] = version("gradio")
    except PackageNotFoundError:
        pass
    return v


def provenance(backend, cfg: dict) -> dict:
    d = cfg["decoding"]
    return {
        "app": "TrialSpec-LM screening prototype (Task 6)",
        **backend.info,
        "decoding": {"strategy": "greedy", "do_sample": False, "num_beams": 1,
                     "repetition_penalty": d.get("repetition_penalty", 1.0), "max_new_tokens": d["max_new_tokens"],
                     "retry_max_new_tokens": d["retry_max_new_tokens"], "eos": ["<|im_end|>", "<|endoftext|>"],
                     "batch_size": 1, "fallback": "constrained label scoring (sum log-prob of each allowed answer)"},
        "aggregation_rule": RULE_TEXT,
        "config": cfg.get("config_path"),
        "git_commit": git_commit(),
        "versions": _versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def screen(prep: Prepared, backend, cfg: dict, progress=None) -> Screening:
    """Run every criterion in order. progress(i, n, criterion) is called before each one."""
    if not prep.check.ok:
        raise ValueError("inputs did not pass validation: " + " ".join(prep.check.errors))
    results = []
    n = len(prep.criteria)
    for i, c in enumerate(prep.criteria):
        if progress:
            progress(i, n, c)
        results.append(run_criterion(backend, prep.sentences, c, prep.prompts[c.id], cfg["decoding"]))
    prov = provenance(backend, cfg)
    prov["total_seconds"] = round(sum(r["seconds"] for r in results), 1)
    return Screening(prep.sentences, prep.split_mode, results, aggregate(results), prov,
                     list(prep.check.warnings), list(prep.notices))
