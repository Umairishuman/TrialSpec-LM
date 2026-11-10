"""Gradio interface (presentation only; all logic lives in pipeline/bench/report)."""
import traceback
from html import escape

import gradio as gr
import pandas as pd

from src.app.bench import MODES, T4_REFERENCE, chat_prompt, run_benchmark
from src.app.pipeline import prepare, screen
from src.app.render import (banner_html, bench_html, cited_by, messages_html, outcome_html, provenance_line,
                            results_html, sentences_html)
from src.app.report import write_reports
from src.app.text import parse_criteria, split_note

ALL = "All criteria"
CUSTOM, FROM_SCREEN = "Custom text prompt", "Criterion from the Screen tab"
NOTE_PLACEHOLDER = ("Paste a patient note. Free text is split into numbered sentences automatically; a note already "
                    "in the dataset format (\"0. ...\" per line) keeps its sentence IDs.")


def _gold_map(example: dict) -> dict:
    return {(c["type"], c["text"]): c["label"] for c in example.get("expert", [])}


def _active_gold(state: dict | None, note_text: str) -> dict:
    """Expert labels apply only while the loaded example's note is unchanged (criteria are matched by text)."""
    return state["map"] if state and state.get("note") == note_text else {}


def _about(cfg: dict, info: dict) -> str:
    lim = cfg["limits"]
    return f"""
### What this app does
For each criterion, the fine-tuned model reads the numbered patient note and returns a label plus the IDs of the
sentences that support it (`{{"evidence_ids": [...], "label": "..."}}`). The explanation is then **built
deterministically from the cited sentences** (D-009), so every statement points back to the note.
Criteria are processed one by one (about 10–15 s each on this laptop's CPU, mostly reading the ~300-token prompt).

**Model:** `{info['model_id']}` + LoRA adapter `{info['adapter']}`, prompt {info['prompt_version']}.
Source of this choice: {info['adapter_source']}. Decoding: greedy, repetition penalty 1.0, at most
{cfg['decoding']['max_new_tokens']} new tokens. These are the settings used in the evaluation.

### Labels and flags
| Flag | Meaning |
|---|---|
| 🟠 missing information | label "not enough information": the note does not say enough |
| ⚪ no supporting evidence cited | a definite label with no cited sentence; it rests on the *absence* of information (expert convention, D-016) |
| 🔴 invalid IDs dropped | the model cited sentence IDs that do not exist in the note |
| 🔁 retried / fallback | the answer was cut off (retried with a larger budget) or unparseable (label chosen by likelihood ranking; counted as unresolved) |

### Overall outcome (rule-based aggregation, D-029)
The dataset labels single criteria, not whole patients, so the overall result is **an explicit rule applied to
the model's predictions, not a ground-truth label**:
1. any blocking criterion (inclusion "not included" / exclusion "excluded") → **Not eligible (preliminary)**;
2. otherwise any unresolved criterion (not enough information, fallback, error), or no inclusion criterion met → **Needs review**;
3. otherwise → **Likely eligible (preliminary)**. "Not applicable" criteria are ignored.

### Input limits
Note ≤ {lim['max_note_chars']:,} characters and ≤ {lim['max_sentences']} sentences; ≤ {lim['max_criteria']} criteria;
each criterion ≤ {lim['max_criterion_chars']} characters; each full prompt ≤ {lim['max_prompt_tokens']} tokens
(the training length; warning above {lim['warn_prompt_tokens']}).

### Start-up
`python -m src.app` → http://localhost:7860 (models are loaded before the page opens).
`python -m src.app --prefetch` downloads every model once so the demo runs offline. Configuration: `configs/app.yaml`.

*{"Educational screening prototype — not a clinical enrollment decision."}*
"""


def build_ui(backend, cfg: dict, examples: list[dict]) -> gr.Blocks:
    limits, report_dir = cfg["limits"], cfg["report_dir"]
    bcfg = cfg["benchmark"]
    by_name = {e["name"]: e for e in examples}

    with gr.Blocks(title="TrialSpec-LM screening", analytics_enabled=False) as demo:
        gr.HTML(banner_html())
        gr.HTML(provenance_line(backend.info))
        gold_state = gr.State({})
        screen_state = gr.State(None)

        # ------------------------------------------------------------------ Screen tab
        with gr.Tab("Screen"):
            with gr.Row(equal_height=True):
                ex_dd = gr.Dropdown([e["name"] for e in examples], label="Example trial (train/val split only; never test)",
                                    value=None, scale=4)
                load_btn = gr.Button("Load example", scale=1)
                clear_btn = gr.Button("Clear", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    note = gr.Textbox(label="Patient note", lines=12, max_lines=24, placeholder=NOTE_PLACEHOLDER)
                with gr.Column(scale=1):
                    gr.Markdown("**Numbered sentences** (what the model sees)")
                    preview = gr.HTML(sentences_html([]))
            with gr.Row():
                incl = gr.Textbox(label="Inclusion criteria (one per line)", lines=6, max_lines=20)
                excl = gr.Textbox(label="Exclusion criteria (one per line)", lines=6, max_lines=20)
            run_btn = gr.Button("Run screening", variant="primary")
            msgs = gr.HTML()
            outcome = gr.HTML()
            table = gr.HTML()
            with gr.Row(equal_height=True):
                hl = gr.Dropdown([ALL], value=ALL, label="Highlight evidence in the note for", scale=3)
                show_raw = gr.Checkbox(label="Show raw model output", value=False, scale=1)
            note_view = gr.HTML()
            with gr.Row():
                md_dl = gr.DownloadButton("Download report (Markdown)", visible=False)
                html_dl = gr.DownloadButton("Download report (HTML, printable to PDF)", visible=False)

        # ------------------------------------------------------------------ Benchmark tab
        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Run the implemented decoding mechanisms **live** on the same prompt and compare them. "
                "*Uncached* recomputes the whole sequence for every new token (reference); the two *KV cache* modes "
                "reuse stored keys/values (Task 3); *speculative* lets the 0.5B draft propose k tokens that the "
                "target checks in one pass (Task 5). All modes are greedy, so their tokens should be identical. "
                "On this CPU the uncached mode costs about one full prompt pass (~5 s) per token, so keep its "
                "token count small.")
            with gr.Row(equal_height=True):
                src = gr.Radio([FROM_SCREEN, CUSTOM], value=FROM_SCREEN, label="Prompt", scale=2)
                crit_dd = gr.Dropdown([], label="Criterion (uses the note from the Screen tab)", scale=3)
            custom = gr.Textbox(label="Custom prompt", lines=3, visible=False,
                                value="In two sentences, what does a clinical trial exclusion criterion do?")
            modes = gr.CheckboxGroup([(v, k) for k, v in MODES.items()], value=list(MODES), label="Decoding modes")
            with gr.Row():
                max_new = gr.Slider(1, bcfg["max_new_tokens_limit"], value=bcfg["default_max_new_tokens"], step=1,
                                    label="Max new tokens")
                k = gr.Slider(1, 8, value=bcfg["default_k"], step=1, label="Speculative k (draft tokens per round)")
                repeats = gr.Slider(1, 3, value=1, step=1, label="Timed repeats (median reported)")
            with gr.Row():
                warm = gr.Checkbox(value=True, label="Warm-up run (2 tokens) before timing")
                eos = gr.Checkbox(value=True, label="Stop at the end-of-answer token")
            bench_btn = gr.Button("Run benchmark", variant="primary")
            bench_msg = gr.HTML()
            bench_table = gr.HTML()
            with gr.Row():
                plot_lat = gr.BarPlot(None, x="mode", y="seconds", title="End-to-end latency (s), measured now",
                                      sort=None, y_title="seconds")
                plot_tps = gr.BarPlot(None, x="mode", y="tokens per second", title="Throughput (tok/s), measured now",
                                      sort=None)
            bench_out = gr.HTML()
            ref_rows = "\n".join(f"| {a} | {b} |" for a, b in T4_REFERENCE)
            gr.Markdown("**Reference only (not live):** Kaggle T4 fp16 results from the project benchmarks "
                        "(docs/runs.md attempt 9).\n\n| Setting | Result |\n|---|---|\n" + ref_rows)

        with gr.Tab("About"):
            gr.Markdown(_about(cfg, backend.info))

        # ------------------------------------------------------------------ handlers
        def on_note(text):
            sentences, mode = split_note(text)
            return sentences_html(sentences, split_mode=mode)

        def crit_choices(i_text, e_text):
            inc, _ = parse_criteria(i_text)
            exc, _ = parse_criteria(e_text)
            ch = [f"I{n}: {t[:90]}" for n, t in enumerate(inc, 1)] + [f"E{n}: {t[:90]}" for n, t in enumerate(exc, 1)]
            return gr.update(choices=ch, value=ch[0] if ch else None)

        def on_load(name):
            if not name:
                gr.Warning("Choose an example first.")
                return gr.update(), gr.update(), gr.update(), {}
            e = by_name[name]
            return e["note"], "\n".join(e["inclusion"]), "\n".join(e["exclusion"]), {"note": e["note"], "map": _gold_map(e)}

        def on_clear():
            return ("", "", "", {}, None, "", "", "", "", gr.update(choices=[ALL], value=ALL),
                    gr.update(visible=False, value=None), gr.update(visible=False, value=None))

        def render_note(s, which):
            if s is None:
                return ""
            hits = cited_by(s.results)
            if which and which != ALL:
                cid = which.split(":")[0]
                hits = {i: c for i, c in hits.items() if cid in c}
            return sentences_html(s.sentences, set(hits), cited_by(s.results), s.split_mode)

        def on_run(note_text, i_text, e_text, gold_st, raw, progress=gr.Progress()):
            empty = ("", "", "", gr.update(choices=[ALL], value=ALL), None,
                     gr.update(visible=False), gr.update(visible=False))
            gold = _active_gold(gold_st, note_text)
            try:
                prep = prepare(note_text, i_text, e_text, backend, limits)
                if not prep.check.ok:
                    return (messages_html(prep.check.errors, prep.check.warnings, prep.notices),) + empty

                def cb(i, n, c):
                    progress(i / n, desc=f"Criterion {i + 1}/{n} ({c.id}): {c.text[:60]}")
                s = screen(prep, backend, cfg, cb)
                md, html = write_reports(s, report_dir, gold)
                n_fb = sum(r["status"] in ("fallback", "error") for r in s.results)
                info = [f"Screened {len(s.results)} criteria in {s.provenance['total_seconds']} s."]
                if n_fb:
                    info.append(f"{n_fb} criterion/criteria needed the fallback or failed; they count as unresolved.")
                choices = [ALL] + [f"{r['id']}: {r['criterion_text'][:70]}" for r in s.results]
                return (messages_html([], s.warnings, s.notices + info), outcome_html(s.outcome),
                        results_html(s.results, gold, raw), render_note(s, ALL),
                        gr.update(choices=choices, value=ALL), s,
                        gr.update(value=str(md), visible=True), gr.update(value=str(html), visible=True))
            except Exception as ex:
                traceback.print_exc()
                return (messages_html([f"Something went wrong while screening: {type(ex).__name__}: {ex}. "
                                       "Your inputs are kept; please try again or shorten them."]),) + empty

        def on_rerender(s, raw, gold_st, note_text):
            return "" if s is None else results_html(s.results, _active_gold(gold_st, note_text), raw)

        def on_bench(source, crit_label, custom_text, note_text, i_text, e_text, modes_sel, n_new, kk, rep, w, stop,
                     progress=gr.Progress()):
            blank = ("", None, None, "")
            try:
                if source == CUSTOM:
                    if not (custom_text or "").strip():
                        return (messages_html(["The custom prompt is empty."]),) + blank
                    prompt = chat_prompt(backend.tok, custom_text)
                else:
                    sentences, _ = split_note(note_text)
                    if not sentences:
                        return (messages_html(["Enter a patient note on the Screen tab first, or use a custom prompt."]),) + blank
                    if not crit_label:
                        return (messages_html(["Enter at least one criterion on the Screen tab, or use a custom prompt."]),) + blank
                    cid = crit_label.split(":")[0]
                    inc, _ = parse_criteria(i_text)
                    exc, _ = parse_criteria(e_text)
                    ctype, items = ("inclusion", inc) if cid.startswith("I") else ("exclusion", exc)
                    idx = int(cid[1:]) - 1
                    if idx >= len(items):
                        return (messages_html(["The criteria changed; pick the criterion again."]),) + blank
                    prompt = backend.prompt(sentences, ctype, items[idx])
                n_tok = backend.count_tokens(prompt)
                if n_tok > limits["max_prompt_tokens"]:
                    return (messages_html([f"The prompt has {n_tok} tokens; the limit is {limits['max_prompt_tokens']}."]),) + blank
                if not modes_sel:
                    return (messages_html(["Select at least one decoding mode."]),) + blank
                order = [m for m in MODES if m in modes_sel]

                def cb(j, n, m):
                    progress(j / n, desc=f"Running {MODES[m]} ({j + 1}/{n})")
                b = run_benchmark(backend, prompt, order, int(n_new), int(kk), bool(w), int(rep), bool(stop), cb)
                df = pd.DataFrame({"mode": [r["label"] for r in b["rows"]],
                                   "seconds": [round(r["total_s"], 3) for r in b["rows"]],
                                   "tokens per second": [round(r["tok_per_s"], 2) for r in b["rows"]]})
                outs = "".join(f"<div class='ts-ev-line'><b>{MODES[m]}:</b> <span class='ts-raw'>{escape(t)}</span></div>"
                               for m, t in b["texts"].items())
                ok = all(r["same_tokens_as_reference"] for r in b["rows"])
                note_ = ("All modes produced identical tokens." if ok else
                         "Some modes differ from the reference; see the table (a near-tie in the logits can flip a "
                         "greedy choice in low precision, D-019/D-023).")
                return (messages_html(notices=[note_]), bench_html(b), df, df, "<b>Generated text per mode</b>" + outs)
            except Exception as ex:
                traceback.print_exc()
                return (messages_html([f"Benchmark failed: {type(ex).__name__}: {ex}"]),) + blank

        note.change(on_note, note, preview)
        for box in (incl, excl):
            box.change(crit_choices, [incl, excl], crit_dd)
        src.change(lambda s: (gr.update(visible=s == CUSTOM), gr.update(visible=s == FROM_SCREEN)), src, [custom, crit_dd])
        load_btn.click(on_load, ex_dd, [note, incl, excl, gold_state])
        clear_btn.click(on_clear, None, [note, incl, excl, gold_state, screen_state, msgs, outcome, table, note_view,
                                         hl, md_dl, html_dl])
        run_btn.click(on_run, [note, incl, excl, gold_state, show_raw],
                      [msgs, outcome, table, note_view, hl, screen_state, md_dl, html_dl])
        hl.change(render_note, [screen_state, hl], note_view)
        show_raw.change(on_rerender, [screen_state, show_raw, gold_state, note], table)
        bench_btn.click(on_bench, [src, crit_dd, custom, note, incl, excl, modes, max_new, k, repeats, warm, eos],
                        [bench_msg, bench_table, plot_lat, plot_tps, bench_out])
    return demo
