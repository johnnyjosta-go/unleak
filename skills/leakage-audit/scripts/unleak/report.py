"""Render findings as text, JSON, SARIF, GitHub annotations or Markdown."""

from __future__ import annotations

import json
import os
from collections import Counter

from . import __version__, rules
from .checker import FileError, Finding

HOMEPAGE = "https://github.com/johnnyjosta-go/unleak"
RULES_URL = HOMEPAGE + "/blob/main/docs/rules.md#{anchor}"

COLORS = {
    "error": "\033[31m", "warning": "\033[33m", "info": "\033[34m",
    "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m", "underline": "\033[4m",
    "green": "\033[32m",
}


def _c(text: str, *styles: str, color: bool) -> str:
    if not color:
        return text
    return "".join(COLORS[s] for s in styles) + text + COLORS["reset"]


def _loc(f: Finding) -> str:
    if f.line <= 0:
        return "-"
    if f.cell is not None:
        return f"cell {f.cell}:{f.cell_line}"
    return f"{f.line}:{f.column}"


def _rel_loc(rel: dict) -> str:
    where = f"cell {rel['cell']}:{rel['cell_line']}" if rel.get("cell") is not None else f"line {rel['line']}"
    if rel.get("path"):
        return f"{rel['path']} {where}"
    return where


def summary_counts(findings: list[Finding]) -> Counter:
    return Counter(f.severity for f in findings)


def text(findings: list[Finding], errors: list[FileError], files_checked: int,
         color: bool = False, snippets: bool = True) -> str:
    out: list[str] = []
    by_path: dict[str, list[Finding]] = {}
    for f in findings:
        by_path.setdefault(f.path, []).append(f)
    for path, items in by_path.items():
        out.append(_c(path, "bold", "underline", color=color))
        width = max(len(_loc(f)) for f in items)
        for f in items:
            sev = _c(f"{f.severity:<7}", f.severity, color=color)
            out.append(f"  {_loc(f):<{width}}  {sev}  {_c(f.rule, 'bold', color=color)}  {f.message}")
            pad = " " * (width + 2 + 2 + 7 + 2 + len(f.rule) + 2)
            for rel in f.related:
                out.append(_c(f"{pad}↳ {_rel_loc(rel)}: {rel['message']}", "dim", color=color))
            if snippets and f.snippet:
                out.append(_c(f"{pad}│ {f.snippet}", "dim", color=color))
        out.append("")
    for e in errors:
        out.append(_c(f"{e.path}: skipped ({e.message})", "dim", color=color))
    if errors:
        out.append("")
    counts = summary_counts(findings)
    if findings:
        parts = [f"{counts[s]} {s}{'s' if counts[s] != 1 else ''}" for s in ("error", "warning", "info") if counts[s]]
        n_files = len(by_path)
        line = f"✖ {len(findings)} problem{'s' if len(findings) != 1 else ''} ({', '.join(parts)}) " \
               f"in {n_files} file{'s' if n_files != 1 else ''}"
        out.append(_c(line, "bold", "error" if counts["error"] else "warning", color=color))
        first = findings[0].rule
        out.append(_c(f"  Run `unleak explain {first}` to see why it matters and how to fix it.", "dim", color=color))
    else:
        out.append(_c(f"✔ No leakage found in {files_checked} file{'s' if files_checked != 1 else ''}.",
                      "bold", "green", color=color))
    return "\n".join(out) + "\n"


def as_json(findings: list[Finding], errors: list[FileError], files_checked: int) -> str:
    counts = summary_counts(findings)
    payload = {
        "tool": "unleak",
        "version": __version__,
        "files_checked": files_checked,
        "summary": {s: counts[s] for s in ("error", "warning", "info")},
        "findings": [dict(f.to_dict(), docs=docs_url(f.rule)) for f in findings],
        "errors": [{"path": e.path, "message": e.message} for e in errors],
    }
    return json.dumps(payload, indent=2) + "\n"


def docs_url(rule_id: str) -> str:
    r = rules.get(rule_id)
    anchor = f"{r.id}-{r.name}".lower()
    return RULES_URL.format(anchor=anchor)


def _uri(path: str) -> str:
    rel = os.path.relpath(path) if os.path.isabs(path) else path
    return rel.replace(os.sep, "/")


def sarif(findings: list[Finding], errors: list[FileError]) -> str:
    all_rules = list(rules.RULES.values())
    index = {r.id: i for i, r in enumerate(all_rules)}
    level = {"error": "error", "warning": "warning", "info": "note"}
    results = []
    for f in findings:
        msg = f.message
        line = f.line if f.line > 0 else 1
        if f.cell is not None:
            msg = f"[cell {f.cell}, line {f.cell_line}] {msg}"
            line = 1
        result = {
            "ruleId": f.rule,
            "ruleIndex": index[f.rule],
            "level": level[f.severity],
            "message": {"text": msg},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": _uri(f.path)},
                    "region": {"startLine": line, "startColumn": max(f.column, 1)},
                }
            }],
        }
        if f.related:
            result["relatedLocations"] = [{
                "id": i,
                "message": {"text": rel["message"]},
                "physicalLocation": {
                    "artifactLocation": {"uri": _uri(rel.get("path") or f.path)},
                    "region": {"startLine": rel["line"] if rel.get("cell") is None else 1},
                },
            } for i, rel in enumerate(f.related)]
        results.append(result)
    driver_rules = [{
        "id": r.id,
        "name": r.name,
        "shortDescription": {"text": r.title},
        "fullDescription": {"text": r.why},
        "help": {"text": r.fix, "markdown": f"{r.fix}\n\n**Bad**\n```python\n{r.bad}\n```\n\n**Good**\n```python\n{r.good}\n```"},
        "helpUri": docs_url(r.id),
        "defaultConfiguration": {"level": level[r.severity]},
        "properties": {"tags": ["data-leakage", "machine-learning", "correctness"]},
    } for r in all_rules]
    run = {
        "tool": {"driver": {"name": "unleak", "version": __version__, "informationUri": HOMEPAGE,
                            "rules": driver_rules}},
        "results": results,
        "invocations": [{
            "executionSuccessful": True,
            "toolExecutionNotifications": [
                {"level": "warning", "message": {"text": f"{e.path}: {e.message}"}} for e in errors
            ],
        }],
    }
    return json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }, indent=2) + "\n"


def github(findings: list[Finding], errors: list[FileError]) -> str:
    """GitHub Actions workflow commands: annotations on the PR diff."""
    level = {"error": "error", "warning": "warning", "info": "notice"}

    def esc(s: str) -> str:
        return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    def esc_prop(s: str) -> str:
        return esc(s).replace(":", "%3A").replace(",", "%2C")

    out = []
    for f in findings:
        line = f.line if f.cell is None and f.line > 0 else 1
        msg = f.message if f.cell is None else f"[cell {f.cell}, line {f.cell_line}] {f.message}"
        out.append(f"::{level[f.severity]} file={esc_prop(_uri(f.path))},line={line},col={max(f.column, 1)},"
                   f"title={esc_prop(f.rule + ' ' + rules.get(f.rule).title)}::{esc(msg)}")
    for e in errors:
        out.append(f"::notice file={esc_prop(_uri(e.path))}::{esc('unleak skipped this file: ' + e.message)}")
    return "\n".join(out) + ("\n" if out else "")


def markdown(findings: list[Finding], errors: list[FileError], files_checked: int) -> str:
    if not findings:
        return f"### ✅ unleak: no data leakage found in {files_checked} file(s)\n"
    counts = summary_counts(findings)
    head = ", ".join(f"{counts[s]} {s}{'s' if counts[s] != 1 else ''}" for s in ("error", "warning", "info") if counts[s])
    icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}
    lines = [f"### unleak: {len(findings)} potential leak(s) ({head})", "",
             "| | Rule | Location | Problem |", "|---|---|---|---|"]
    for f in findings:
        where = f"`{_uri(f.path)}` {_loc(f)}"
        rule = f"[{f.rule}]({docs_url(f.rule)})"
        message = f.message.replace("|", "\\|")
        lines.append(f"| {icon[f.severity]} | {rule} | {where} | {message} |")
    lines.append("")
    lines.append("Run `unleak explain <RULE>` for the reasoning and a fix.")
    return "\n".join(lines) + "\n"


def rules_table(fmt: str = "text") -> str:
    all_rules = list(rules.RULES.values())
    if fmt == "json":
        return json.dumps([r.__dict__ for r in all_rules], indent=2) + "\n"
    if fmt == "markdown":
        out = ["| Rule | Severity | Name | What it catches |", "|---|---|---|---|"]
        for r in all_rules:
            anchor = f"{r.id}-{r.name}".lower()
            out.append(f"| [{r.id}](#{anchor}) | {r.severity} | `{r.name}` | {r.title} |")
        return "\n".join(out) + "\n"
    width = max(len(r.name) for r in all_rules)
    return "\n".join(f"{r.id}  {r.severity:<7}  {r.name:<{width}}  {r.title}" for r in all_rules) + "\n"


def explain(rule_id: str, color: bool = False) -> str:
    r = rules.get(rule_id)
    indent = lambda s: "\n".join("    " + line for line in s.splitlines())  # noqa: E731
    return "\n".join([
        _c(f"{r.id} {r.name}", "bold", color=color) + f"  ({_c(r.severity, r.severity, color=color)})",
        r.title,
        "",
        _c("Why it matters", "bold", color=color),
        r.why,
        "",
        _c("How to fix", "bold", color=color),
        r.fix,
        "",
        _c("Leaky", "bold", "error", color=color),
        indent(r.bad),
        "",
        _c("Fixed", "bold", "green", color=color),
        indent(r.good),
        "",
        f"Suppress a deliberate case with `# unleak: ignore[{r.id}]` on the reported line.",
        "",
    ])
