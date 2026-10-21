"""Run the analyzer over files and turn raw findings into reportable ones."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from . import engine, rules, source

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".ipynb_checkpoints", "node_modules",
    ".venv", "venv", "env", ".env", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages", "build", "dist", ".eggs",
}


@dataclass
class Finding:
    rule: str
    name: str
    severity: str
    path: str
    line: int
    column: int
    end_line: int
    message: str
    title: str
    cell: int | None = None
    cell_line: int | None = None
    snippet: str = ""
    related: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def display_location(self) -> str:
        if self.cell is not None:
            return f"cell {self.cell}, line {self.cell_line}"
        return f"{self.line}:{self.column}"


@dataclass
class FileError:
    path: str
    message: str


def check_source(src: source.Source) -> list[Finding]:
    if src.tree is None or src.skip:
        return []
    raw = engine.analyze(src.tree)
    out = []
    for r in raw:
        rule = rules.get(r.rule)
        line = getattr(r.node, "lineno", 1)
        end = getattr(r.node, "end_lineno", line) or line
        if src.suppressed(line, rule.id) or (end != line and src.suppressed(end, rule.id)):
            continue
        cell, cell_line = src.location(line)
        related = []
        for rel_line, text in r.related:
            rc, rl = src.location(rel_line)
            related.append({"line": rel_line, "cell": rc, "cell_line": rl, "message": text})
        out.append(Finding(
            rule=rule.id, name=rule.name, severity=rule.severity, path=src.path,
            line=line, column=getattr(r.node, "col_offset", 0) + 1, end_line=end,
            message=r.message, title=rule.title, cell=cell,
            cell_line=cell_line if cell is not None else None,
            snippet=src.line_text(line).strip(), related=related,
        ))
    out.sort(key=lambda f: (f.line, f.column, f.rule))
    return out


def check_text(text: str, path: str = "<string>") -> list[Finding]:
    return check_source(source.load(path, text))


def check_file(path: str) -> tuple[list[Finding], FileError | None]:
    try:
        src = source.load(path)
    except OSError as exc:
        return [], FileError(path, str(exc))
    if src.parse_error:
        return [], FileError(path, src.parse_error)
    return check_source(src), None


def iter_files(paths: list[str]):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
                for f in sorted(files):
                    if f.endswith((".py", ".ipynb")):
                        yield os.path.join(root, f)
        else:
            yield p
