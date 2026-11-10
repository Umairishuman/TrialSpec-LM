"""Start the screening app:  python -m src.app   ->  http://localhost:7860

  python -m src.app                    load the configured models and serve the app
  python -m src.app --prefetch         download every model/adapter once (then the demo works offline)
  python -m src.app --smoke            headless end-to-end check on tiny random models (no download, ~1 min)
  python -m src.app --smoke --launch   same tiny models, but serve the UI (for UI testing without the real model)
"""
import argparse
import sys
import time
from pathlib import Path

from src.app.config import DEFAULT_CONFIG, load_app_config
from src.utils import load_config, load_env

SMOKE_NOTE = ("A 64-year-old man presents with chest pain on exertion for two weeks. He has type 2 diabetes "
              "treated with metformin 500 mg b.i.d. and a history of hypertension.\n"
              "Dr. Smith noted an HbA1c of 7.8%. He denies any history of stroke.")


def prefetch(cfg: dict) -> None:
    from huggingface_hub import snapshot_download
    repos = [cfg["model_id"], cfg["draft_model"], cfg.get("adapter"), cfg["resolved"]["adapter"]]
    fe = Path(cfg["final_eval_config"])
    if fe.exists():   # every final-eval candidate, so the final selection works offline too
        repos += [c["adapter"] for c in load_config(fe).get("candidates", [])]
    for repo in dict.fromkeys(r for r in repos if r and not Path(r).exists()):
        t0 = time.perf_counter()
        snapshot_download(repo)
        print(f"ok  {repo}  ({time.perf_counter() - t0:.0f} s)")
    print("prefetch complete: the app can now start offline (set HF_HUB_OFFLINE=1 to be sure).")


def smoke(cfg: dict, launch: bool) -> None:
    from src.app.backend import ModelBackend
    from src.app.bench import MODES, run_benchmark
    from src.app.examples import load_examples
    from src.app.pipeline import prepare, screen
    from src.app.report import write_reports
    from src.app.ui import build_ui

    t0 = time.perf_counter()
    backend = ModelBackend.tiny(cfg["resolved"]["prompt_version"])
    cfg = {**cfg, "decoding": {**cfg["decoding"], "max_new_tokens": 8, "retry_max_new_tokens": 12}}

    bad = prepare("", "", "", backend, cfg["limits"])
    assert not bad.check.ok and len(bad.check.errors) == 2, bad.check.errors
    print("validation: empty note and empty criteria rejected:", *bad.check.errors, sep="\n  ")

    prep = prepare(SMOKE_NOTE, "- Type 2 diabetes", "History of stroke", backend, cfg["limits"])   # 1 note, 2 criteria
    assert prep.check.ok, prep.check.errors
    print(f"split ({prep.split_mode}): {len(prep.sentences)} sentences; criteria: {[c.id for c in prep.criteria]}")
    for i, s in prep.sentences:
        print(f"  {i}. {s}")
    s = screen(prep, backend, cfg, lambda i, n, c: print(f"  criterion {i + 1}/{n}: {c.id}"))
    for r in s.results:
        print(f"  {r['id']}: label={r['label']!r} status={r['status']} attempts={len(r['attempts'])} "
              f"raw={r['attempts'][-1]['raw'][:40]!r}")
        assert r["label"] is not None, "fallback must always produce an allowed label"
    print(f"outcome: {s.outcome.verdict} | {s.outcome.reasons}")
    md, html = write_reports(s, Path(cfg["report_dir"]) / "smoke")
    assert "not a clinical enrollment decision" in md.read_text(encoding="utf-8")
    print(f"reports: {md} , {html}")

    b = run_benchmark(backend, prep.prompts["I1"], list(MODES), max_new_tokens=6, k=2, warmup=True, repeats=1,
                      stop_at_eos=False)
    for r in b["rows"]:
        print(f"  {r['label']:38s} total={r['total_s']:.3f}s tokens={r['new_tokens']} same={r['same_tokens_as_reference']}")
    assert all(r["same_tokens_as_reference"] for r in b["rows"]), "decoders disagree in fp32"

    examples = load_examples(cfg["examples"])
    demo = build_ui(backend, cfg, examples)
    print(f"UI built ({len(examples)} examples). Smoke test passed in {time.perf_counter() - t0:.1f} s.")
    if launch:
        serve(demo, cfg)


def serve(demo, cfg: dict, open_browser: bool = False) -> None:
    from src.app.render import CSS
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    print(f"\nOpen http://localhost:{port} in a browser (Ctrl+C to stop).\n")
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, css=CSS,
                                                   inbrowser=open_browser, show_error=True)


def main():
    ap = argparse.ArgumentParser(description="TrialSpec-LM screening app (Task 6)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true", help="headless check on tiny random models")
    ap.add_argument("--launch", action="store_true", help="with --smoke: serve the UI on the tiny models")
    ap.add_argument("--prefetch", action="store_true", help="download all models and exit")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--open", action="store_true", help="open the browser when ready")
    args = ap.parse_args()
    load_env()
    for stream in (sys.stdout, sys.stderr):      # Windows consoles (cp1252) cannot print arbitrary model output
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    cfg = load_app_config(args.config)
    if args.port:
        cfg["server"]["port"] = args.port
    if args.host:
        cfg["server"]["host"] = args.host
    r = cfg["resolved"]
    print(f"model: {cfg['model_id']} + {r['adapter']} (prompt {r['prompt_version']}); source: {r['source']}")

    if args.prefetch:
        return prefetch(cfg)
    if args.smoke:
        return smoke(cfg, args.launch)

    from src.app.backend import ModelBackend
    from src.app.examples import load_examples
    from src.app.ui import build_ui
    try:
        backend = ModelBackend.load(cfg)
    except OSError as e:
        sys.exit(f"Could not load the model ({e}).\nIf this machine is offline, run `python -m src.app --prefetch` "
                 "once while online.")
    serve(build_ui(backend, cfg, load_examples(cfg["examples"])), cfg, args.open)


if __name__ == "__main__":
    main()
