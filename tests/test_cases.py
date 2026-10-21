"""Every file in tests/cases declares the findings it must produce.

A finding is expected where a line carries `# expect: UL101` (several ids may be
comma-separated). Files without markers must produce no findings at all.
"""

import pathlib
import re

import pytest

from unleak.checker import check_file

CASES_DIR = pathlib.Path(__file__).parent / "cases"
CASES = sorted(CASES_DIR.glob("*.py"))
EXPECT = re.compile(r"#\s*expect:\s*([A-Z0-9,\s]+)$")


def expected(path: pathlib.Path) -> set[tuple[int, str]]:
    out = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        m = EXPECT.search(line)
        if m:
            out.update((lineno, rid.strip()) for rid in m.group(1).split(",") if rid.strip())
    return out


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.name)
def test_case(path: pathlib.Path) -> None:
    findings, error = check_file(str(path))
    assert error is None, error.message
    got = {(f.line, f.rule) for f in findings}
    want = expected(path)
    missing = want - got
    unexpected = got - want
    details = "\n".join(f"  line {f.line}: {f.rule} {f.message}" for f in findings)
    assert not missing and not unexpected, (
        f"missing={sorted(missing)} unexpected={sorted(unexpected)}\nfindings:\n{details}"
    )
