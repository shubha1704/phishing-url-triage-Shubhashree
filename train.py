"""Train, tune and save the five phishing classifiers.

Usage (BITS Virtual Lab):  put the Kaggle/UCI CSV in data/ and run  python train.py
Outputs: model/*.joblib, model/meta.json, test_data.csv, results_table.md
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import phish_core as core

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 3110
TEST_ROWS = 10_000        # held-out file; small enough for the free Streamlit tier
SLOW_MODEL_ROWS = 30_000  # SVM / KNN are trained on a stratified sample of the training set
ROOT = Path(__file__).parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "model"


def load_dataset() -> pd.DataFrame:
    hits = sorted(DATA_DIR.glob("*.csv"))
    if not hits:
        sys.exit(f"No CSV found in {DATA_DIR}. Put PhiUSIIL_Phishing_URL_Dataset.csv there.")
    df = pd.read_csv(hits[0])
    df.columns = [c.strip() for c in df.columns]
    if core.RAW_TARGET not in df.columns:
        sys.exit(f"Column '{core.RAW_TARGET}' missing. Found: {list(df.columns)[-3:]}")
    return df


def candidates():
    def pipe(clf, scale):
        steps = [("feat", FunctionTransformer(core.make_features, validate=False))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("clf", clf)])

    return {
        "Logistic Regression": (pipe(LogisticRegression(max_iter=4000, random_state=SEED), True),
                                {"clf__C": [0.1, 1, 10]}, False),
        "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED), False),
                          {"clf__max_depth": [6, 10, 16], "clf__min_samples_leaf": [2, 10, 30],
                           "clf__criterion": ["gini", "entropy"]}, False),
        "KNN": (pipe(KNeighborsClassifier(n_jobs=-1), True),
                {"clf__n_neighbors": [5, 15, 31], "clf__weights": ["uniform", "distance"]}, True),
        "Naive Bayes": (pipe(GaussianNB(), False),
                        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}, False),
        "Support Vector Machine": (
            pipe(CalibratedClassifierCV(SVC(kernel="rbf"), cv=3, ensemble=False), True),
            {"clf__estimator__C": [1, 10], "clf__estimator__gamma": ["scale", 0.01]}, True),
    }


def main():
    t0 = time.time()
    df = load_dataset()
    y = core.to_binary(df[core.RAW_TARGET])
    print(f"{len(df):,} rows, {df.shape[1] - 1} raw columns | phishing share {y.mean():.1%}")

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_ROWS, stratify=y,
                                      random_state=SEED)
    train, test = df.iloc[idx_tr], df.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    test.to_csv(ROOT / "test_data.csv", index=False)  # raw columns incl. URL and `label`

    # stratified sample for the distance/kernel models
    sub_idx, _ = train_test_split(np.arange(len(train)), train_size=SLOW_MODEL_ROWS,
                                  stratify=y_tr, random_state=SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows, best = [], {}
    for name, (pipe, grid, slow) in candidates().items():
        t = time.time()
        Xf, yf = (train.iloc[sub_idx], y_tr[sub_idx]) if slow else (train, y_tr)
        gs = GridSearchCV(pipe, grid, scoring="matthews_corrcoef", cv=cv, n_jobs=-1).fit(Xf, yf)
        model = gs.best_estimator_
        proba = model.predict_proba(test)[:, 1]
        m = core.evaluate(y_te, (proba >= 0.5).astype(int), proba)
        rows.append({"ML Model Name": name, **m})
        best[name] = {k.split("__", 1)[1]: v for k, v in gs.best_params_.items()}
        joblib.dump(model, MODEL_DIR / core.MODEL_FILES[name], compress=3)
        print(f"{name:24s} {time.time() - t:6.1f}s trained on {len(Xf):,} rows  {best[name]}")

    res = pd.DataFrame(rows).set_index("ML Model Name")
    print("\n" + res.round(4).to_string())
    md = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|---|"]
    md += [f"| {n} | " + " | ".join(f"{r[c]:.4f}" for c in core.METRIC_ORDER) + " |" for n, r in res.iterrows()]
    (ROOT / "results_table.md").write_text("\n".join(md) + "\n")

    meta = {"class_names": core.CLASS_NAMES, "n_train": int(len(train)), "n_test": int(len(test)),
            "slow_model_rows": SLOW_MODEL_ROWS, "best_params": best, "seed": SEED,
            "raw_columns": [c for c in df.columns if c != core.RAW_TARGET],
            "versions": {"scikit-learn": sklearn.__version__}}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nBest MCC: {res['MCC'].idxmax()} | pin scikit-learn=={sklearn.__version__} in requirements.txt "
          f"| {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
