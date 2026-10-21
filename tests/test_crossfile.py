from unleak.checker import check_paths

PREPARE = '''import pandas as pd
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/raw.csv")
X, y = SMOTE().fit_resample(df.drop(columns=["label"]), df["label"])
pd.concat([X, y], axis=1).to_csv("../data/balanced.csv", index=False)
'''
TRAIN = '''import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/balanced.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["label"]), df["label"])
'''


def test_leak_saved_in_one_file_and_split_in_another(tmp_path):
    (tmp_path / "prep").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "prep" / "prepare.py").write_text(PREPARE)
    (tmp_path / "model" / "train.py").write_text(TRAIN)
    findings, errors, files = check_paths([str(tmp_path)])
    assert files == 2 and not errors
    assert [(f.rule, f.path.endswith("prepare.py"), f.line) for f in findings] == [("UL103", True, 5)]
    rel = findings[0].related[0]
    assert rel["path"].endswith("train.py") and rel["line"] == 5


def test_unrelated_file_names_do_not_link(tmp_path):
    (tmp_path / "prepare.py").write_text(PREPARE.replace("../data/balanced.csv", "balanced_v1.csv"))
    (tmp_path / "train.py").write_text(TRAIN.replace("../data/balanced.csv", "other.csv"))
    findings, _, _ = check_paths([str(tmp_path)])
    assert findings == []


def test_numpy_save_and_load(tmp_path):
    (tmp_path / "a.py").write_text(
        "import numpy as np\nfrom sklearn.preprocessing import StandardScaler\n"
        "X = np.load('raw.npy')\nnp.save('scaled.npy', StandardScaler().fit_transform(X))\n")
    (tmp_path / "b.py").write_text(
        "import numpy as np\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.svm import SVC\n"
        "X = np.load('scaled.npy')\ny = np.load('y.npy')\nprint(cross_val_score(SVC(), X, y))\n")
    findings, _, _ = check_paths([str(tmp_path)])
    assert [(f.rule, f.path.endswith("a.py")) for f in findings] == [("UL101", True)]
