"""Load .py files and Jupyter notebooks into parseable Python source."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

SUPPRESS_RE = re.compile(r"#\s*(?:unleak|noqa)\s*:\s*(?:ignore)?\s*(\[[^\]]*\]|[A-Z]{2}\d{3}(?:\s*,\s*[A-Z]{2}\d{3})*)?", re.IGNORECASE)
SKIP_FILE_RE = re.compile(r"#\s*unleak\s*:\s*skip-file", re.IGNORECASE)

# Cell magics whose body is still Python.
PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun", "debug", "memit", "skip"}


@dataclass
class Source:
    path: str
    text: str
    is_notebook: bool = False
    # For notebooks: global line number (1-based) -> (cell number, line in cell)
    cell_lines: dict[int, tuple[int, int]] = field(default_factory=dict)
    suppressions: dict[int, set[str] | None] = field(default_factory=dict)
    skip: bool = False
    parse_error: str | None = None
    tree: ast.Module | None = None

    def line_text(self, line: int) -> str:
        lines = self.text.splitlines()
        if 1 <= line <= len(lines):
            return lines[line - 1]
        return ""

    def location(self, line: int) -> tuple[int | None, int]:
        """(cell, line) for notebooks, (None, line) for scripts."""
        if self.is_notebook and line in self.cell_lines:
            return self.cell_lines[line]
        return None, line

    def suppressed(self, line: int, rule_id: str) -> bool:
        if line not in self.suppressions:
            return False
        ids = self.suppressions[line]
        return ids is None or any(rule_id.startswith(i) for i in ids)


def _sanitize_cell(lines: list[str]) -> list[str]:
    """Replace IPython syntax with blank lines, keeping line numbers stable."""
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines[0].lstrip()[2:].split()[0] if lines[0].lstrip()[2:].split() else ""
        if magic in PYTHON_CELL_MAGICS:
            return [""] + [_sanitize_line(line) for line in lines[1:]]
        return ["" for _ in lines]
    return [_sanitize_line(line) for line in lines]


def _sanitize_line(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith(("%", "!")):
        return indent + "pass" if indent else ""
    if stripped.endswith("?") and not stripped.startswith("#") and "=" not in stripped:
        return ""
    # `x = !ls` and `x = %sql ...`
    m = re.match(r"^(\s*[\w\.,\s]+=\s*)[!%]", line)
    if m:
        return m.group(1) + "None"
    return line


def _suppressions(text: str) -> dict[int, set[str] | None]:
    out: dict[int, set[str] | None] = {}
    for i, line in enumerate(text.splitlines(), start=1):
        if "#" not in line:
            continue
        m = SUPPRESS_RE.search(line)
        if not m:
            continue
        if "noqa" in m.group(0).lower() and not m.group(1):
            continue  # a bare noqa comment belongs to other linters
        ids = m.group(1)
        if ids:
            parts = [p.strip().upper() for p in ids.strip("[]").split(",") if p.strip()]
            out[i] = set(parts) if parts else None
        else:
            out[i] = None
    return out


def load_notebook(path: str, raw: str) -> Source:
    nb = json.loads(raw)
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out_lines: list[str] = []
    cell_lines: dict[int, tuple[int, int]] = {}
    cell_no = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        cell_no += 1
        src = cell.get("source", cell.get("input", ""))
        if isinstance(src, list):
            src = "".join(src)
        lines = _sanitize_cell(src.splitlines())
        # Drop cells that do not parse on their own (e.g. half-written code).
        try:
            ast.parse("\n".join(lines))
        except SyntaxError:
            lines = ["" for _ in lines]
        for j, line in enumerate(lines, start=1):
            out_lines.append(line)
            cell_lines[len(out_lines)] = (cell_no, j)
        out_lines.append("")  # separator keeps cells apart
        cell_lines[len(out_lines)] = (cell_no, len(lines) + 1)
    text = "\n".join(out_lines) + "\n"
    return Source(path=path, text=text, is_notebook=True, cell_lines=cell_lines)


def load(path: str, text: str | None = None) -> Source:
    if text is None:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    if path.endswith(".ipynb"):
        if not text.strip():
            return Source(path=path, text="", is_notebook=True, tree=ast.parse(""))
        try:
            src = load_notebook(path, text)
        except (ValueError, KeyError, TypeError) as exc:
            return Source(path=path, text="", is_notebook=True, parse_error=f"invalid notebook: {exc}")
    else:
        src = Source(path=path, text=text)
    src.skip = bool(SKIP_FILE_RE.search(src.text))
    src.suppressions = _suppressions(src.text)
    try:
        src.tree = ast.parse(src.text, filename=path)
    except SyntaxError as exc:
        src.parse_error = f"syntax error at line {exc.lineno}: {exc.msg}"
    return src
