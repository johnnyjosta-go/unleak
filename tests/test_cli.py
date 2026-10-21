import io
import json
import pathlib

import pytest

from unleak import cli, hook, rules
from unleak.checker import check_text

CASES = pathlib.Path(__file__).parent / "cases"


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_check_text_and_exit_code(capsys):
    code, out = run(capsys, "check", str(CASES / "ul103_smote.py"), "--color", "never")
    assert code == 1
    assert "UL103" in out and "error" in out
    assert "↳ line 7: rows are split into train/test here" in out


def test_clean_file_exits_zero(capsys):
    code, out = run(capsys, str(CASES / "clean_pipeline_cv.py"), "--color", "never")
    assert code == 0
    assert "No leakage found" in out


def test_json_format(capsys):
    code, out = run(capsys, "check", str(CASES / "ul301_target_not_dropped.py"), "-f", "json")
    payload = json.loads(out)
    assert payload["summary"]["error"] == 1
    f = payload["findings"][0]
    assert f["rule"] == "UL301" and f["line"] == 7 and f["docs"].endswith("#ul301-target-in-features")


def test_sarif_format(capsys):
    _, out = run(capsys, "check", str(CASES / "ul101_scaler_before_split.py"), "-f", "sarif")
    sarif = json.loads(out)
    assert sarif["version"] == "2.1.0"
    run0 = sarif["runs"][0]
    assert len(run0["tool"]["driver"]["rules"]) == len(rules.RULES)
    res = run0["results"][0]
    assert res["ruleId"] == "UL101" and res["level"] == "warning"
    assert run0["tool"]["driver"]["rules"][res["ruleIndex"]]["id"] == "UL101"
    assert res["locations"][0]["physicalLocation"]["region"]["startLine"] == 10


def test_github_format(capsys):
    _, out = run(capsys, "check", str(CASES / "ul201_fit_transform_test.py"), "-f", "github")
    assert out.startswith("::error file=")
    assert "title=UL201" in out


def test_markdown_format(capsys):
    _, out = run(capsys, "check", str(CASES / "ul103_smote.py"), "-f", "markdown")
    assert "| 🔴 | [UL103]" in out


def test_fail_on_and_min_severity(capsys):
    path = str(CASES / "ul101_scaler_before_split.py")  # a warning
    assert run(capsys, "check", path, "--fail-on", "error")[0] == 0
    assert run(capsys, "check", path)[0] == 1
    code, out = run(capsys, "check", path, "--min-severity", "error")
    assert code == 0 and "UL101" not in out


def test_select_and_ignore(capsys):
    path = str(CASES / "ul106_fillna_mean.py")
    assert run(capsys, "check", path, "--ignore", "UL1")[0] == 0
    assert run(capsys, "check", path, "--select", "UL3")[0] == 0
    assert run(capsys, "check", path, "--select", "UL106")[0] == 1


def test_info_hidden_by_default(capsys):
    path = str(CASES / "ul403_entity_ids.py")
    code, out = run(capsys, "check", path)
    assert code == 0 and "UL403" not in out
    code, out = run(capsys, "check", path, "--min-severity", "info", "--fail-on", "info")
    assert code == 1 and "UL403" in out


def test_rules_and_explain(capsys):
    _, out = run(capsys, "rules", "-f", "markdown")
    assert out.count("\n") >= len(rules.RULES) + 2
    code, out = run(capsys, "explain", "ul102", "--color", "never")
    assert code == 0 and "Leaky" in out and "Fixed" in out
    code, out = run(capsys, "explain", "target-in-features", "--color", "never")
    assert code == 0 and "UL301" in out
    assert cli.main(["explain", "UL999"]) == 2


def test_directory_walk_skips_venv(tmp_path, capsys):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text((CASES / "ul103_smote.py").read_text())
    (tmp_path / "ok.py").write_text("print('hi')\n")
    code, out = run(capsys, "check", str(tmp_path), "--color", "never")
    assert code == 0 and "1 file" in out


def test_syntax_error_is_reported_not_raised(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n")
    code, out = run(capsys, "check", str(bad), "--color", "never")
    assert code == 0 and "skipped" in out


def test_suppression_variants():
    base = "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\n"
    leak = "X = StandardScaler().fit_transform(X){}\nX_tr, X_te = train_test_split(X)\n"
    assert len(check_text(base + leak.format(""))) == 1
    assert check_text(base + leak.format("  # unleak: ignore")) == []
    assert check_text(base + leak.format("  # unleak: ignore[UL101]")) == []
    assert check_text(base + leak.format("  # noqa: UL101")) == []
    assert len(check_text(base + leak.format("  # unleak: ignore[UL3]"))) == 1
    assert check_text("# unleak: skip-file\n" + base + leak.format("")) == []


def test_hook_reports_findings(tmp_path):
    f = tmp_path / "train.py"
    f.write_text((CASES / "ul103_smote.py").read_text())
    event = {"tool_name": "Write", "tool_input": {"file_path": str(f)}, "cwd": str(tmp_path)}
    out = io.StringIO()
    assert hook.main(io.StringIO(json.dumps(event)), out) == 0
    payload = json.loads(out.getvalue())
    assert payload["decision"] == "block"
    assert "UL103" in payload["reason"] and "fix:" in payload["reason"]


@pytest.mark.parametrize("event", [
    {"tool_name": "Write", "tool_input": {"file_path": "/nope/missing.py"}},
    {"tool_name": "Edit", "tool_input": {"file_path": "README.md"}},
    {"tool_name": "Bash", "tool_input": {"command": "ls"}},
    "not json",
])
def test_hook_is_silent_otherwise(event):
    out = io.StringIO()
    raw = event if isinstance(event, str) else json.dumps(event)
    assert hook.main(io.StringIO(raw), out) == 0
    assert out.getvalue() == ""


def test_hook_clean_file_is_silent(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text((CASES / "clean_fit_on_train.py").read_text())
    out = io.StringIO()
    hook.main(io.StringIO(json.dumps({"tool_input": {"file_path": str(f)}})), out)
    assert out.getvalue() == ""


def test_hook_can_be_disabled(tmp_path, monkeypatch):
    f = tmp_path / "train.py"
    f.write_text((CASES / "ul103_smote.py").read_text())
    monkeypatch.setenv("UNLEAK_HOOK", "off")
    out = io.StringIO()
    hook.main(io.StringIO(json.dumps({"tool_input": {"file_path": str(f)}})), out)
    assert out.getvalue() == ""
