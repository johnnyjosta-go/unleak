import json

from unleak.checker import check_file
from unleak.source import load


def make_nb(path, cells):
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src, **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, src in cells
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))
    return path


def test_notebook_findings_map_to_cells(tmp_path):
    nb = make_nb(tmp_path / "leaky.ipynb", [
        ("markdown", "# Fraud model"),
        ("code", "%matplotlib inline\n!pip install imbalanced-learn\nimport pandas as pd\n"
                 "from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import train_test_split"),
        ("code", "df = pd.read_csv('fraud.csv')\nX = df.drop(columns=['is_fraud'])\ny = df['is_fraud']"),
        ("code", "%%time\nX_res, y_res = SMOTE().fit_resample(X, y)"),
        ("code", "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)\nX_train.shape?"),
        ("code", "%%bash\necho this is not python ("),
    ])
    findings, err = check_file(str(nb))
    assert err is None
    assert [(f.rule, f.cell, f.cell_line) for f in findings] == [("UL103", 3, 2)]
    assert findings[0].related[0]["cell"] == 4


def test_broken_cell_does_not_hide_the_rest(tmp_path):
    nb = make_nb(tmp_path / "half.ipynb", [
        ("code", "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split"),
        ("code", "def oops(:\n    pass"),
        ("code", "X = StandardScaler().fit_transform(X)\nX_tr, X_te = train_test_split(X)"),
    ])
    findings, err = check_file(str(nb))
    assert err is None
    assert [(f.rule, f.cell) for f in findings] == [("UL101", 3)]


def test_source_list_and_empty_notebook(tmp_path):
    nb = make_nb(tmp_path / "list.ipynb", [("code", ["import numpy as np\n", "x = np.ones(3)\n"])])
    src = load(str(nb))
    assert src.tree is not None and "np.ones" in src.text
    empty = tmp_path / "empty.ipynb"
    empty.write_text("")
    findings, err = check_file(str(empty))
    assert findings == [] and err is None


def test_invalid_notebook_is_an_error(tmp_path):
    bad = tmp_path / "bad.ipynb"
    bad.write_text("{not json")
    findings, err = check_file(str(bad))
    assert findings == [] and "invalid notebook" in err.message
