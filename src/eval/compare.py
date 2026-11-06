"""Markdown comparison table of *.metrics.json files (for the run log and the report)."""
import argparse
import json
from pathlib import Path


def row(name: str, path: Path) -> str:
    m = json.loads(path.read_text())
    r, me = m["metrics"], m["meta"]
    return (f"| {name} | {me.get('prompt_version') or 'v1'} | {r['overall']['accuracy']:.3f} | {r['overall']['macro_f1']:.3f} | "
            f"{r['inclusion']['macro_f1']:.3f} | {r['exclusion']['macro_f1']:.3f} | {r['evidence']['micro']['f1']:.3f} | "
            f"{r['evidence']['macro']['f1']:.3f} | {me['parse_ok_rate']:.0%} | {me['git_commit'][:7]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+", help="name=path/to/file.metrics.json")
    args = ap.parse_args()
    print("| Model | Prompt | Acc | Macro-F1 | Incl. mF1 | Excl. mF1 | Evid. micro-F1 | Evid. macro-F1 | Parse OK | Commit |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e in args.entries:
        name, path = e.split("=", 1)
        print(row(name, Path(path)))


if __name__ == "__main__":
    main()
