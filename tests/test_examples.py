"""Each example's leaky pipeline is flagged with its rule; the honest one is not."""

import ast
import pathlib

import pytest

from unleak.checker import check_file

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"
FILES = sorted(p for p in EXAMPLES.glob("*.py") if p.name != "run_all.py" and not p.name.startswith("demo_"))


def functions(path):
    tree = ast.parse(path.read_text())
    return {n.name: (n.lineno, n.end_lineno) for n in tree.body if isinstance(n, ast.FunctionDef)}


def declared_rule(path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RULE":
            return node.value.value
    raise AssertionError(f"{path.name} has no RULE")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_example_is_flagged_only_in_leaky(path):
    findings, err = check_file(str(path))
    assert err is None
    funcs = functions(path)
    start, end = funcs["leaky"]
    assert [f.rule for f in findings] == [declared_rule(path)], [(f.rule, f.line, f.message) for f in findings]
    assert all(start <= f.line <= end for f in findings)
