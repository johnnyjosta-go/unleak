"""Claude Code PostToolUse hook: lint ML files right after the agent writes them.

Reads the hook event from stdin. When the edited file has leakage findings it
prints `{"decision": "block", "reason": ...}` so Claude sees them next to the
tool result and can fix the code before reporting any metric. It never fails
the tool call: any internal problem exits 0 silently.
"""

from __future__ import annotations

import json
import os
import sys

from . import rules
from .checker import check_file

MAX_FINDINGS = 8


def message_for(path: str, findings) -> str:
    shown = findings[:MAX_FINDINGS]
    lines = [f"unleak found possible data leakage in {os.path.basename(path)}:"]
    for f in shown:
        where = f"cell {f.cell}, line {f.cell_line}" if f.cell is not None else f"line {f.line}"
        rel = f" ({f.related[0]['message']} at line {f.related[0]['line']})" if f.related else ""
        lines.append(f"- {where} [{f.rule} {f.severity}] {f.message}{rel}")
        lines.append(f"  fix: {rules.get(f.rule).fix}")
    if len(findings) > len(shown):
        lines.append(f"- ... and {len(findings) - len(shown)} more (run `unleak check {path}`)")
    lines.append(
        "Fix these before trusting or reporting any metric from this code. `unleak explain <RULE>` "
        "has details. If a finding is a deliberate, justified choice, add `# unleak: ignore[<RULE>]` "
        "on that line and tell the user why."
    )
    return "\n".join(lines)


def main(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if os.environ.get("UNLEAK_HOOK", "").lower() in ("0", "off", "false", "no"):
        return 0
    try:
        event = json.load(stdin)
        tool_input = event.get("tool_input") or {}
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not path or not path.endswith((".py", ".ipynb")):
            return 0
        if not os.path.isabs(path) and event.get("cwd"):
            path = os.path.join(event["cwd"], path)
        if not os.path.isfile(path):
            return 0
        findings, error = check_file(path)
        findings = [f for f in findings if f.severity in ("error", "warning")]
        if error is not None or not findings:
            return 0
        json.dump({"decision": "block", "reason": message_for(path, findings)}, stdout)
        stdout.write("\n")
    except Exception:  # never break the agent's tool call
        return 0
    return 0
