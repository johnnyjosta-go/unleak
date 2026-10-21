import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from unleak import cli  # noqa: E402
from unleak.data import check  # noqa: E402


def frames(seed=0, n=400):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": rng.integers(0, 120, n),
        "age": rng.integers(18, 90, n),
        "bp": rng.normal(120, 15, n).round(1),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "outcome": rng.integers(0, 2, n),
    })
    return df


def rules_of(findings):
    return sorted(f.rule for f in findings)


def test_clean_split_has_no_findings():
    df = frames()
    train, test = df.iloc[:300], df.iloc[300:]
    test = test[~test["patient_id"].isin(train["patient_id"])]
    assert check("train.csv", "test.csv", target="outcome", group="patient_id", time="date",
                 train=train, test=test) == []


def test_duplicates_groups_and_time():
    df = frames()
    shuffled = df.sample(frac=1, random_state=1)
    train, test = shuffled.iloc[:300], shuffled.iloc[300:]
    test = pd.concat([test, train.iloc[:5]])  # copied rows
    found = check("train.csv", "test.csv", group="patient_id", time="date", train=train, test=test)
    assert rules_of(found) == ["UD001", "UD002", "UD003"]
    dup = next(f for f in found if f.rule == "UD001")
    assert dup.message.startswith("5 of")


def test_proxy_column_binary_and_regression():
    df = frames()
    df["days_in_collections"] = df["outcome"] * 30 + np.random.default_rng(2).integers(0, 3, len(df))
    found = check("t.csv", "s.csv", target="outcome", train=df.iloc[:300], test=df.iloc[300:])
    assert [f.message.split("'")[1] for f in found if f.rule == "UD004"] == ["days_in_collections"]

    df["price"] = df["bp"] * 3.5 + 10
    df["price_x2"] = df["price"] * 2
    found = check("t.csv", "s.csv", target="price", train=df.iloc[:300], test=df.iloc[300:])
    cols = sorted(f.message.split("'")[1] for f in found if f.rule == "UD004")
    assert cols == ["bp", "price_x2"]


def test_cli_data_command(tmp_path, capsys):
    df = frames()
    train, test = df.iloc[:300], pd.concat([df.iloc[300:], df.iloc[:3]])
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False) if _has_parquet() else test.to_csv(tmp_path / "test.csv", index=False)
    test_path = tmp_path / ("test.parquet" if _has_parquet() else "test.csv")
    code = cli.main(["data", str(tmp_path / "train.csv"), str(test_path), "--group", "patient_id", "--color", "never"])
    out = capsys.readouterr().out
    assert code == 1
    assert "UD001" in out and "UD002" in out


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False
