"""HTML fragments shared by the UI and the exported report. Every user/model string is escaped."""
from html import escape

from src.app.aggregate import LIKELY_ELIGIBLE, NEEDS_REVIEW, NOT_ELIGIBLE

DISCLAIMER = "Educational screening prototype — not a clinical enrollment decision."

CSS = """
.ts-banner{background:rgba(220,38,38,.12);border:1px solid rgba(220,38,38,.55);border-radius:8px;padding:10px 14px;
  font-weight:600;margin:4px 0}
.ts-prov{font-size:.85em;opacity:.85;margin:2px 0 6px}
.ts-msg{border-radius:8px;padding:8px 12px;margin:6px 0}
.ts-msg.err{background:rgba(220,38,38,.12);border:1px solid rgba(220,38,38,.5)}
.ts-msg.warn{background:rgba(217,119,6,.12);border:1px solid rgba(217,119,6,.5)}
.ts-msg.info{background:rgba(37,99,235,.10);border:1px solid rgba(37,99,235,.4)}
.ts-msg ul{margin:4px 0 0 18px;padding:0}
.ts-outcome{border-radius:10px;padding:12px 16px;margin:8px 0;border:2px solid}
.ts-outcome.not{border-color:rgba(220,38,38,.7);background:rgba(220,38,38,.08)}
.ts-outcome.review{border-color:rgba(217,119,6,.75);background:rgba(217,119,6,.08)}
.ts-outcome.likely{border-color:rgba(22,163,74,.7);background:rgba(22,163,74,.08)}
.ts-outcome .v{font-size:1.35em;font-weight:700}
.ts-outcome .k{font-size:.8em;text-transform:uppercase;letter-spacing:.05em;opacity:.8}
.ts-outcome ul{margin:6px 0 0 18px;padding:0}
.ts-rule{font-size:.82em;opacity:.8;margin-top:6px}
table.ts{border-collapse:collapse;width:100%;font-size:.92em}
table.ts th,table.ts td{border-bottom:1px solid rgba(127,127,127,.3);padding:6px 8px;text-align:left;vertical-align:top}
table.ts th{font-weight:600;background:rgba(127,127,127,.08)}
table.ts tr.unresolved td{background:rgba(217,119,6,.07)}
table.ts td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.ts-badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.85em;font-weight:600;white-space:nowrap;border:1px solid}
.ts-badge.satisfied{background:rgba(22,163,74,.13);border-color:rgba(22,163,74,.6)}
.ts-badge.blocking{background:rgba(220,38,38,.13);border-color:rgba(220,38,38,.6)}
.ts-badge.unresolved{background:rgba(217,119,6,.15);border-color:rgba(217,119,6,.65)}
.ts-badge.neutral{background:rgba(127,127,127,.12);border-color:rgba(127,127,127,.5)}
.ts-flag{display:inline-block;font-size:.8em;margin:1px 0;white-space:nowrap}
mark.ts-ev{background:rgba(250,204,21,.35);color:inherit;border-radius:3px;padding:0 2px}
.ts-ev-line{margin:2px 0}
.ts-note{font-size:.93em;line-height:1.5}
.ts-note .s{padding:2px 6px;border-radius:4px;margin:1px 0}
.ts-note .s.hit{background:rgba(250,204,21,.30)}
.ts-note .sid{display:inline-block;min-width:2.2em;font-weight:600;opacity:.7}
.ts-chip{display:inline-block;font-size:.75em;border:1px solid rgba(127,127,127,.5);border-radius:6px;padding:0 5px;margin-left:4px}
.ts-raw{font-family:monospace;font-size:.8em;opacity:.75;white-space:pre-wrap;word-break:break-word}
"""

def banner_html() -> str:
    return f'<div class="ts-banner">⚠ {escape(DISCLAIMER)}</div>'


def provenance_line(info: dict) -> str:
    return (f'<div class="ts-prov">Model: <b>{escape(str(info["model_id"]))}</b> + adapter '
            f'<b>{escape(str(info["adapter"]))}</b> (prompt {escape(info["prompt_version"])}; source: '
            f'{escape(info["adapter_source"])}) · {escape(info["dtype"])} on {escape(info["device"])} · greedy decoding, '
            f'repetition penalty 1.0</div>')


def messages_html(errors=(), warnings=(), notices=()) -> str:
    parts = []
    for cls, title, items in (("err", "Please fix the input", errors), ("warn", "Warning", warnings),
                              ("info", "Note", notices)):
        if items:
            lis = "".join(f"<li>{escape(m)}</li>" for m in items)
            parts.append(f'<div class="ts-msg {cls}"><b>{title}</b><ul>{lis}</ul></div>')
    return "".join(parts)


def sentences_html(sentences, highlight: set[int] = frozenset(), cited_by: dict[int, list[str]] | None = None,
                   split_mode: str | None = None) -> str:
    if not sentences:
        return '<div class="ts-note"><i>No sentences yet.</i></div>'
    head = ""
    if split_mode == "numbered":
        head = "<div class='ts-prov'>The note already had numbered sentences; their original IDs are kept.</div>"
    elif split_mode == "split":
        head = "<div class='ts-prov'>The note was split automatically into numbered sentences (IDs from 0).</div>"
    rows = []
    for i, s in sentences:
        chips = "".join(f'<span class="ts-chip">{escape(c)}</span>' for c in (cited_by or {}).get(i, []))
        cls = "s hit" if i in highlight else "s"
        rows.append(f'<div class="{cls}"><span class="sid">{i}.</span>{escape(s)}{chips}</div>')
    return f'<div class="ts-note">{head}{"".join(rows)}</div>'


def cited_by(results) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for r in results:
        for i in r["evidence_ids"]:
            out.setdefault(i, []).append(r["id"])
    return out


def outcome_html(outcome) -> str:
    cls = {NOT_ELIGIBLE: "not", NEEDS_REVIEW: "review", LIKELY_ELIGIBLE: "likely"}[outcome.verdict]
    reasons = "".join(f"<li>{escape(r)}</li>" for r in outcome.reasons)
    missing = ""
    if outcome.missing_information:
        missing = (f"<div style='margin-top:6px'>🟠 <b>Missing information</b> for "
                   f"{escape(', '.join(outcome.missing_information))}: the note does not say enough to decide "
                   f"these criteria; a clinician would need to ask for it.</div>")
    return (f'<div class="ts-outcome {cls}"><div class="k">Overall preliminary outcome · rule-based aggregation, '
            f'not ground truth</div><div class="v">{escape(outcome.verdict)}</div><ul>{reasons}</ul>{missing}'
            f'<div class="ts-rule"><b>Rule (D-029):</b> {escape(outcome.rule)}</div></div>')


def _flags(r) -> str:
    f = r["flags"]
    out = []
    if f["missing_information"]:
        out.append("🟠 missing information")
    if f["no_evidence"]:
        out.append("⚪ no supporting evidence cited")
    if f["hallucinated_ids"]:
        out.append(f"🔴 invalid IDs dropped: {', '.join(map(str, r['invalid_ids']))}")
    if r["status"] == "retried":
        out.append("🔁 retried (longer budget)")
    if r["status"] == "fallback":
        out.append("🔁 fallback label")
    if r["status"] == "error":
        out.append("🔴 model error")
    return "<br>".join(f'<span class="ts-flag">{escape(x)}</span>' for x in out) or "—"


def results_html(results, gold: dict | None = None, show_raw: bool = False) -> str:
    gold = gold or {}
    has_gold = any((r["criterion_type"], r["criterion_text"]) in gold for r in results)
    head = ("<tr><th>#</th><th>Criterion</th><th>Label</th><th>Evidence (cited sentences)</th><th>Explanation</th>"
            "<th>Flags</th>" + ("<th>Expert label (example)</th>" if has_gold else "") + "<th>Time</th></tr>")
    rows = []
    for r in results:
        label = r["label"] or "no decision"
        badge = f'<span class="ts-badge {r["aggregate_status"]}">{escape(label)}</span>'
        ev = "".join(f'<div class="ts-ev-line"><mark class="ts-ev">[{c["id"]}]</mark> {escape(c["text"])}</div>'
                     for c in r["evidence"]) or "<i>none cited</i>"
        expl = escape(r["explanation"])
        if show_raw:
            raws = " | ".join(a["raw"] for a in r["attempts"])
            expl += f'<div class="ts-raw">raw: {escape(raws)}</div>'
        g = ""
        if has_gold:
            gl = gold.get((r["criterion_type"], r["criterion_text"]))
            g = "<td>—</td>" if gl is None else f"<td>{escape(gl)} {'✓' if gl == r['label'] else '✗'}</td>"
        rows.append(f'<tr class="{r["aggregate_status"]}"><td><b>{r["id"]}</b><br><small>{r["criterion_type"]}</small></td>'
                    f'<td>{escape(r["criterion_text"])}</td><td>{badge}</td><td>{ev}</td><td>{expl}</td>'
                    f'<td>{_flags(r)}</td>{g}<td class="num">{r["seconds"]:.1f} s</td></tr>')
    return f'<table class="ts">{head}{"".join(rows)}</table>'


def _fmt(x, nd=2, suffix=""):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:.{nd}f}{suffix}"
    return f"{x}{suffix}"


def bench_html(bench: dict) -> str:
    ref = bench["reference_mode"]
    head = ("<tr><th>Mode</th><th>New tokens</th><th>Prefill</th><th>Decode</th><th>Total</th><th>tok/s</th>"
            f"<th>Speed-up vs {escape(ref)}</th><th>Target forwards</th><th>Acceptance</th><th>KV cache</th>"
            "<th>Same tokens as reference</th></tr>")
    rows = []
    for r in bench["rows"]:
        same = "✔" if r["same_tokens_as_reference"] else f"✘ (first difference at token {r['first_diff']})"
        rows.append(
            f"<tr><td>{escape(r['label'])}</td><td class='num'>{r['new_tokens']}</td>"
            f"<td class='num'>{_fmt(r['prefill_s'], 2, ' s')}</td><td class='num'>{_fmt(r['decode_s'], 2, ' s')}</td>"
            f"<td class='num'><b>{_fmt(r['total_s'], 2, ' s')}</b></td><td class='num'>{_fmt(r['tok_per_s'], 2)}</td>"
            f"<td class='num'>{_fmt(r['speedup_vs_reference'], 2, '×')}</td><td class='num'>{r['target_forwards']}</td>"
            f"<td class='num'>{_fmt(r['acceptance_rate'], 2)}</td><td class='num'>{_fmt(r['kv_cache_mib'], 2, ' MiB')}</td>"
            f"<td>{same}</td></tr>")
    s = bench["settings"]
    foot = (f"<div class='ts-prov'>Prompt: {bench['input_tokens']} tokens · max {s['max_new_tokens']} new tokens · "
            f"{'stops at EOS' if s['stop_at_eos'] else 'EOS ignored'} · warm-up {'on' if s['warmup'] else 'off'} · "
            f"median of {s['repeats']} run(s) · {escape(s['dtype'])} on {escape(s['device'])}. "
            "Uncached decoding re-runs the whole sequence at every step, so its cost grows with length; the cached modes "
            "process one new token per step. Speculative decoding is exact (same tokens) but not always faster: "
            "see D-021/D-023.</div>")
    return f'<table class="ts">{head}{"".join(rows)}</table>{foot}'
