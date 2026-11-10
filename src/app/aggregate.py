"""Overall preliminary outcome from criterion-level predictions: an explicit, documented rule (D-029).

The dataset labels criteria, not patient-trial pairs, so this outcome is an aggregation of model
predictions, not a ground-truth label and not a clinical decision.
"""
from dataclasses import dataclass, field

BLOCKING, UNRESOLVED, SATISFIED, NEUTRAL = "blocking", "unresolved", "satisfied", "neutral"

_STATUS = {
    "inclusion": {"included": SATISFIED, "not included": BLOCKING,
                  "not enough information": UNRESOLVED, "not applicable": NEUTRAL},
    "exclusion": {"not excluded": SATISFIED, "excluded": BLOCKING,
                  "not enough information": UNRESOLVED, "not applicable": NEUTRAL},
}

NOT_ELIGIBLE, NEEDS_REVIEW, LIKELY_ELIGIBLE = "Not eligible (preliminary)", "Needs review", "Likely eligible (preliminary)"

RULE_TEXT = (
    "Per criterion: inclusion 'included' and exclusion 'not excluded' = satisfied; inclusion 'not included' "
    "and exclusion 'excluded' = blocking; 'not enough information', unparseable output, fallback labels and "
    "model errors = unresolved; 'not applicable' = neutral (ignored). "
    "Outcome, first match wins: (1) any blocking criterion -> Not eligible (preliminary); "
    "(2) any unresolved criterion, or no inclusion criterion satisfied -> Needs review; "
    "(3) otherwise -> Likely eligible (preliminary). "
    "This is a rule-based aggregation of model predictions (D-029), not ground truth and not a clinical decision."
)


def criterion_status(criterion_type: str, label: str | None, fallback: bool = False) -> str:
    """Map one criterion result to blocking / unresolved / satisfied / neutral."""
    if fallback or label is None:
        return UNRESOLVED
    return _STATUS[criterion_type][label]


@dataclass
class Outcome:
    verdict: str
    reasons: list[str]
    blocking: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    satisfied: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    no_evidence: list[str] = field(default_factory=list)
    rule: str = RULE_TEXT

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def aggregate(results: list[dict]) -> Outcome:
    """results: dicts with keys id, criterion_type, label, fallback (bool), evidence_ids (list)."""
    if not results:
        raise ValueError("cannot aggregate an empty list of criteria")
    groups = {BLOCKING: [], UNRESOLVED: [], SATISFIED: [], NEUTRAL: []}
    for r in results:
        groups[criterion_status(r["criterion_type"], r["label"], r.get("fallback", False))].append(r["id"])
    missing = [r["id"] for r in results if r["label"] == "not enough information"]
    no_ev = [r["id"] for r in results
             if r["label"] not in (None, "not enough information") and not r.get("fallback") and not r.get("evidence_ids")]
    incl_satisfied = [r["id"] for r in results if r["criterion_type"] == "inclusion" and r["id"] in groups[SATISFIED]]

    if groups[BLOCKING]:
        verdict = NOT_ELIGIBLE
        reasons = [f"Blocking: {', '.join(groups[BLOCKING])} (a definite failure decides the outcome)."]
        if groups[UNRESOLVED]:
            reasons.append(f"Also unresolved: {', '.join(groups[UNRESOLVED])}.")
    elif groups[UNRESOLVED] or not incl_satisfied:
        verdict = NEEDS_REVIEW
        reasons = []
        if groups[UNRESOLVED]:
            reasons.append(f"Unresolved: {', '.join(groups[UNRESOLVED])}.")
        if not incl_satisfied:
            reasons.append("No inclusion criterion is confirmed as met.")
    else:
        verdict = LIKELY_ELIGIBLE
        reasons = [f"All {len(groups[SATISFIED])} applicable criteria are satisfied"
                   + (f"; {len(groups[NEUTRAL])} not applicable." if groups[NEUTRAL] else ".")]
    if no_ev:
        reasons.append(f"{len(no_ev)} decision(s) cite no supporting sentence ({', '.join(no_ev)}): they rest on "
                       "the absence of reported information, not on stated evidence.")
    return Outcome(verdict, reasons, groups[BLOCKING], groups[UNRESOLVED], groups[SATISFIED], groups[NEUTRAL],
                   missing, no_ev)
