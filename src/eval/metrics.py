"""Criterion classification and evidence-sentence metrics.

Empty-set rule for evidence: when gold and predicted sets are both empty the record scores
P=R=F1=1 (correctly cites nothing); if exactly one side is empty, the record scores 0.
Micro scores pool TP/FP/FN over all records, so records with both sets empty contribute nothing.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.load import LABELS

INVALID = "<invalid>"


def classification_report(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    pred = [p if p is not None else INVALID for p in pred]
    present = [l for l in labels if l in gold]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        # Macro-F1 over classes present in gold; absent classes are reported per-class as N/A.
        "macro_f1": f1_score(gold, pred, labels=present, average="macro", zero_division=0) if present else None,
        "invalid_rate": sum(x == INVALID for x in pred) / max(len(pred), 1),
        "per_class": {
            l: ({"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} if s[i] else {"support": 0, "note": "N/A (no gold examples)"})
            for i, l in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels + [INVALID],
            "matrix": confusion_matrix(gold, pred, labels=labels + [INVALID]).tolist(),
        },
    }


def evidence_scores(gold: list[list[int]], pred: list[list[int]]) -> dict:
    tp = fp = fn = 0
    ps, rs, fs = [], [], []
    for g, p in zip(gold, pred):
        g, p = set(g), set(p)
        t = len(g & p)
        tp, fp, fn = tp + t, fp + len(p - g), fn + len(g - p)
        if not g and not p:
            pr = rc = f1 = 1.0
        else:
            pr = t / len(p) if p else 0.0
            rc = t / len(g) if g else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        ps.append(pr); rs.append(rc); fs.append(f1)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    n = max(len(gold), 1)
    return {
        "micro": {"precision": mp, "recall": mr, "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0},
        "macro": {"precision": sum(ps) / n, "recall": sum(rs) / n, "f1": sum(fs) / n},
        "n_gold_empty": sum(not g for g in gold),
        "n_pred_empty": sum(not p for p in pred),
        "n_both_empty": sum(not g and not p for g, p in zip(gold, pred)),
    }


def full_report(records: list[dict]) -> dict:
    """records: dicts with criterion_type, gold_label, pred_label, gold_evidence, pred_evidence."""
    out = {}
    groups = {"inclusion": [r for r in records if r["criterion_type"] == "inclusion"],
              "exclusion": [r for r in records if r["criterion_type"] == "exclusion"]}
    for name, rs in groups.items():
        if rs:
            out[name] = classification_report([r["gold_label"] for r in rs], [r["pred_label"] for r in rs], LABELS[name])
    # Overall: inclusion/exclusion label sets differ, so accuracy and macro-F1 are pooled over the union.
    all_labels = LABELS["inclusion"] + [l for l in LABELS["exclusion"] if l not in LABELS["inclusion"]]
    out["overall"] = classification_report([r["gold_label"] for r in records], [r["pred_label"] for r in records], all_labels)
    out["evidence"] = evidence_scores([r["gold_evidence"] for r in records], [r["pred_evidence"] for r in records])
    return out
