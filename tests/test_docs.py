import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_rules_doc_is_up_to_date():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_docs

    assert (ROOT / "docs" / "rules.md").read_text() == gen_docs.render(), "run: python tools/gen_docs.py"


def test_skill_frontmatter():
    text = (ROOT / "skills" / "leakage-audit" / "SKILL.md").read_text()
    head = text.split("---")[1]
    fields = dict(line.split(": ", 1) for line in head.strip().splitlines())
    assert fields["name"] == "leakage-audit"
    assert len(fields["description"]) <= 1024
    for rel in ("references/taxonomy.md", "references/fixes.md", "references/report-template.md",
                "scripts/run_unleak.py"):
        assert (ROOT / "skills" / "leakage-audit" / rel).exists(), rel
