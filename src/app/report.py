"""Downloadable screening report (rubric G4): Markdown and self-contained HTML, both with provenance."""
import json
import re
from html import escape
from pathlib import Path

from src.app.render import (CSS, DISCLAIMER, banner_html, cited_by, outcome_html, results_html,
                            sentences_html)

_PROV_FIELDS = [("Model", "model_id"), ("Adapter", "adapter"), ("Adapter source", "adapter_source"),
                ("Prompt version", "prompt_version"), ("Dtype", "dtype"), ("Device", "device"),
                ("Git commit", "git_commit"), ("Config", "config"), ("Created (UTC)", "created_utc"),
                ("Total model time (s)", "total_seconds")]


def _md(text) -> str:
    """Escape characters that would break a Markdown table cell."""
    return re.sub(r"([|\\`*_\[\]<>])", r"\\\1", str(text)).replace("\n", " ")


def _code(obj) -> str:
    """JSON as an inline code span (no Markdown escaping needed; pipes/backticks cannot occur in these dicts)."""
    return "`" + json.dumps(obj).replace("`", "'").replace("|", "/") + "`"


def report_markdown(s, gold: dict | None = None) -> str:
    p, o = s.provenance, s.outcome
    lines = ["# Eligibility screening report", "", f"> **{DISCLAIMER}**", "",
             "## Overall preliminary outcome (rule-based aggregation, not ground truth)", "",
             f"**{o.verdict}**", ""]
    lines += [f"- {_md(r)}" for r in o.reasons]
    if o.missing_information:
        lines.append(f"- Missing information: {', '.join(o.missing_information)}")
    lines += ["", f"*Rule (D-029):* {_md(o.rule)}", "", "## Criterion-level results", "",
              "| # | Type | Criterion | Label | Evidence | Explanation | Flags |", "|---|---|---|---|---|---|---|"]
    for r in s.results:
        ev = "; ".join(f"[{c['id']}] {c['text']}" for c in r["evidence"]) or "none cited"
        flags = [k.replace("_", " ") for k, v in r["flags"].items() if v]
        lines.append(f"| {r['id']} | {r['criterion_type']} | {_md(r['criterion_text'])} | {r['label'] or 'no decision'} "
                     f"| {_md(ev)} | {_md(r['explanation'])} | {', '.join(flags) or '—'} |")
    if gold:
        matched = [(r, gold.get((r["criterion_type"], r["criterion_text"]))) for r in s.results]
        if any(g for _, g in matched):
            lines += ["", "Expert labels (example trial from the train/val split; shown for comparison only):", ""]
            lines += [f"- {r['id']}: expert *{g}*, model *{r['label']}*" for r, g in matched if g]
    lines += ["", "## Patient note (numbered sentences)", ""]
    lines += [f"- **{i}.** {_md(t)}" for i, t in s.sentences]
    lines += ["", "## Provenance", "", "| Field | Value |", "|---|---|"]
    lines += [f"| {name} | {_md(p.get(key, ''))} |" for name, key in _PROV_FIELDS]
    lines += [f"| Decoding | {_code(p['decoding'])} |", f"| Library versions | {_code(p['versions'])} |", ""]
    if s.warnings:
        lines += ["Warnings at run time:", ""] + [f"- {_md(w)}" for w in s.warnings] + [""]
    return "\n".join(lines)


def report_html(s, gold: dict | None = None) -> str:
    p = s.provenance
    prov_rows = "".join(f"<tr><th>{escape(n)}</th><td>{escape(str(p.get(k, '')))}</td></tr>" for n, k in _PROV_FIELDS)
    prov_rows += f"<tr><th>Decoding</th><td><code>{escape(json.dumps(p['decoding']))}</code></td></tr>"
    prov_rows += f"<tr><th>Library versions</th><td><code>{escape(json.dumps(p['versions']))}</code></td></tr>"
    hits = cited_by(s.results)
    warn = "".join(f"<li>{escape(w)}</li>" for w in s.warnings)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eligibility screening report</title>
<style>
:root{{color-scheme:light dark}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;
  background:Canvas;color:CanvasText;line-height:1.45}}
h1{{font-size:1.5em}} h2{{font-size:1.15em;margin-top:1.6em}}
code{{font-size:.85em;word-break:break-word}}
@media print{{.ts-banner{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
{CSS}
</style></head><body>
<h1>Eligibility screening report</h1>
{banner_html()}
<h2>Overall preliminary outcome</h2>
{outcome_html(s.outcome)}
<h2>Criterion-level results</h2>
{results_html(s.results, gold)}
<h2>Patient note</h2>
<p class="ts-prov">Highlighted sentences are cited as evidence; chips show which criteria cite them.</p>
{sentences_html(s.sentences, set(hits), hits, s.split_mode)}
{f"<h2>Warnings</h2><ul>{warn}</ul>" if warn else ""}
<h2>Provenance</h2>
<table class="ts">{prov_rows}</table>
</body></html>"""


def write_reports(s, out_dir: Path | str, gold: dict | None = None) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = s.provenance["created_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    md, html = out / f"screening_{stamp}.md", out / f"screening_{stamp}.html"
    md.write_text(report_markdown(s, gold), encoding="utf-8")
    html.write_text(report_html(s, gold), encoding="utf-8")
    return md, html
