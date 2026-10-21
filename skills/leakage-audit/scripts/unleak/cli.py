"""Command-line interface: `unleak check`, `data`, `rules`, `explain`, `hook`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__, report, rules
from .checker import check_paths

COMMANDS = {"check", "data", "rules", "explain", "hook"}
SEVERITIES = ("info", "warning", "error")


def load_config(start: str) -> dict:
    """[tool.unleak] from the nearest pyproject.toml, if Python can read TOML."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return {}
    d = os.path.abspath(start if os.path.isdir(start) else os.path.dirname(start) or ".")
    while True:
        cand = os.path.join(d, "pyproject.toml")
        if os.path.isfile(cand):
            try:
                with open(cand, "rb") as fh:
                    return tomllib.load(fh).get("tool", {}).get("unleak", {})
            except (OSError, ValueError):
                return {}
        parent = os.path.dirname(d)
        if parent == d:
            return {}
        d = parent


def _split(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [v for v in str(value).split(",") if v.strip()]


def _use_color(flag: str) -> bool:
    if flag == "always":
        return True
    if flag == "never" or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def cmd_check(args) -> int:
    cfg = load_config(args.paths[0] if args.paths else ".")
    select = _split(args.select) or _split(cfg.get("select"))
    ignore = _split(args.ignore) or _split(cfg.get("ignore"))
    min_sev = args.min_severity or cfg.get("min-severity", "warning")
    fail_on = args.fail_on or cfg.get("fail-on", "warning")
    exclude = _split(cfg.get("exclude"))

    paths = args.paths or ["."]
    for path in paths:
        if not os.path.exists(path):
            print(f"unleak: {path}: no such file or directory", file=sys.stderr)
            return 2
    findings, errors, files = check_paths(paths, exclude)

    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings = [
        f for f in findings
        if order[f.severity] >= order[min_sev]
        and (not select or rules.matches(f.rule, select))
        and not (ignore and rules.matches(f.rule, ignore))
    ]
    out = {
        "text": lambda: report.text(findings, errors, files, color=_use_color(args.color), snippets=not args.no_snippets),
        "json": lambda: report.as_json(findings, errors, files),
        "sarif": lambda: report.sarif(findings, errors),
        "github": lambda: report.github(findings, errors),
        "markdown": lambda: report.markdown(findings, errors, files),
    }[args.format]()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    elif not (args.quiet and not findings):
        sys.stdout.write(out)
    if fail_on == "never":
        return 0
    return 1 if any(order[f.severity] >= order[fail_on] for f in findings) else 0


def cmd_data(args) -> int:
    from . import data

    findings = data.check(args.train, args.test, target=args.target, group=args.group, time=args.time,
                          ignore=_split(args.ignore_cols))
    if args.format == "json":
        sys.stdout.write(report.as_json(findings, [], 2))
    elif args.format == "markdown":
        sys.stdout.write(report.markdown(findings, [], 2))
    else:
        color = _use_color(args.color)
        if not findings:
            sys.stdout.write(report._c("✔ No data-level leakage found.", "bold", "green", color=color) + "\n")
        for f in findings:
            sev = report._c(f"{f.severity:<7}", f.severity, color=color)
            sys.stdout.write(f"{sev}  {report._c(f.rule, 'bold', color=color)}  {f.message}\n")
    return 1 if any(f.severity == "error" for f in findings) else 0


def cmd_rules(args) -> int:
    sys.stdout.write(report.rules_table(args.format))
    return 0


def cmd_explain(args) -> int:
    rule_id = args.rule.upper()
    if rule_id not in rules.RULES:
        matches = [r for r in rules.RULES.values() if r.name == args.rule.lower()]
        if not matches:
            print(f"unleak: unknown rule '{args.rule}'. Run `unleak rules` for the list.", file=sys.stderr)
            return 2
        rule_id = matches[0].id
    sys.stdout.write(report.explain(rule_id, color=_use_color(args.color)))
    return 0


def cmd_hook(args) -> int:
    from . import hook

    return hook.main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unleak",
        description="Find data leakage in machine-learning code (Python scripts and Jupyter notebooks).",
        epilog="Examples:\n  unleak check train.py notebooks/\n  unleak check . --format sarif -o unleak.sarif\n"
               "  unleak data train.csv test.csv --target price --group customer_id\n  unleak explain UL103",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"unleak {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("check", help="statically analyse .py and .ipynb files (default command)")
    c.add_argument("paths", nargs="*", help="files or directories (default: .)")
    c.add_argument("--format", "-f", choices=["text", "json", "sarif", "github", "markdown"], default="text")
    c.add_argument("--output", "-o", help="write the report to a file")
    c.add_argument("--select", help="only these rules or prefixes, e.g. UL1,UL301")
    c.add_argument("--ignore", help="skip these rules or prefixes")
    c.add_argument("--min-severity", choices=SEVERITIES, help="hide findings below this level (default: warning)")
    c.add_argument("--fail-on", choices=SEVERITIES + ("never",), help="exit 1 at this level or above (default: warning)")
    c.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    c.add_argument("--no-snippets", action="store_true", help="do not print source lines")
    c.add_argument("--quiet", "-q", action="store_true", help="print nothing when no leakage is found")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("data", help="check a real train/test pair: duplicates, group/time overlap, proxy columns")
    d.add_argument("train")
    d.add_argument("test")
    d.add_argument("--target", help="target column: enables the single-feature proxy check")
    d.add_argument("--group", help="entity id column that must not appear on both sides")
    d.add_argument("--time", help="timestamp column: training rows must predate test rows")
    d.add_argument("--ignore-cols", help="comma-separated columns to leave out of the duplicate check")
    d.add_argument("--format", "-f", choices=["text", "json", "markdown"], default="text")
    d.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    d.set_defaults(func=cmd_data)

    r = sub.add_parser("rules", help="list all rules")
    r.add_argument("--format", "-f", choices=["text", "markdown", "json"], default="text")
    r.set_defaults(func=cmd_rules)

    e = sub.add_parser("explain", help="explain a rule with a leaky and a fixed example")
    e.add_argument("rule")
    e.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    e.set_defaults(func=cmd_explain)

    h = sub.add_parser("hook", help="Claude Code PostToolUse hook (reads the event from stdin)")
    h.set_defaults(func=cmd_hook)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["check", "."]
    elif argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv = ["check"] + argv
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except BrokenPipeError:  # pragma: no cover - e.g. `unleak check . | head`
        return 0
