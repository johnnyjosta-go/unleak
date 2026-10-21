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


@dataclass
class FileError:
    path: str
    message: str


def _convert(raw: list, analysed: str, sources: dict[str, source.Source]) -> list[Finding]:
    out = []
    for r in raw:
        rule = rules.get(r.rule)
        path = r.path or analysed
        src = sources.get(path)
        if src is None:
            continue
        line = getattr(r.node, "lineno", 1)
        end = getattr(r.node, "end_lineno", line) or line
        if src.suppressed(line, rule.id) or (end != line and src.suppressed(end, rule.id)):
            continue
        cell, cell_line = src.location(line)
        related = []
        for rel in r.related:
            rel_line, text = rel[0], rel[1]
            rel_path = (rel[2] if len(rel) > 2 else None) or analysed
            rel_src = sources.get(rel_path, src)
            rc, rl = rel_src.location(rel_line)
            entry = {"line": rel_line, "cell": rc, "cell_line": rl, "message": text}
            if rel_path != path:
                entry["path"] = rel_path
            related.append(entry)
        out.append(Finding(
            rule=rule.id, name=rule.name, severity=rule.severity, path=path,
            line=line, column=getattr(r.node, "col_offset", 0) + 1, end_line=end,
            message=r.message, title=rule.title, cell=cell,
            cell_line=cell_line if cell is not None else None,
            snippet=src.line_text(line).strip(), related=related,
        ))
    return out


def check_source(src: source.Source) -> list[Finding]:
    if src.tree is None or src.skip:
        return []
    an = engine.analyze(src.tree, path=src.path)
    found = _convert(an.findings, src.path, {src.path: src})
    found.sort(key=lambda f: (f.line, f.column, f.rule))
    return found


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


def check_paths(paths: list[str], exclude: list[str] | None = None) -> tuple[list[Finding], list[FileError], int]:
    """Check many files at once. Data written by one file (df.to_csv, np.save)
    and loaded by another carries its preprocessing history across files."""
    exclude = exclude or []
    sources: dict[str, source.Source] = {}
    errors: list[FileError] = []
    files = 0
    for path in iter_files(paths):
        if any(part in path for part in exclude):
            continue
        files += 1
        try:
            src = source.load(path)
        except OSError as exc:
            errors.append(FileError(path, str(exc)))
            continue
        if src.parse_error:
            errors.append(FileError(path, src.parse_error))
            continue
        if src.tree is not None and not src.skip:
            sources[path] = src

    analyses = {p: engine.analyze(src.tree, path=p) for p, src in sources.items()}

    written: list[engine.Written] = [w for an in analyses.values() for w in an.writes]
    if written:
        for p, src in sources.items():
            external: dict[str, list] = {}
            for w in written:
                if w.source == p:
                    continue
                full = os.path.normpath(os.path.join(os.path.dirname(w.source or ""), w.path))
                external.setdefault(full, []).append(w)
                external.setdefault(os.path.basename(w.path), []).append(w)
            if external:
                analyses[p] = engine.analyze(src.tree, path=p, external=external)

    findings: list[Finding] = []
    seen = set()
    for p, an in analyses.items():
        for f in _convert(an.findings, p, sources):
            key = (f.rule, f.path, f.line)
            if key not in seen:
                seen.add(key)
                findings.append(f)
    findings.sort(key=lambda f: (f.path, f.line, f.column, f.rule))
    return findings, errors, files


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
