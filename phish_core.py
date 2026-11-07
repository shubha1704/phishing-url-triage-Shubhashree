"""Shared logic for train.py and app.py (features, metrics, cost model).

Label convention of the raw dataset: label 1 = legitimate, label 0 = phishing.
Internally we flip it so that PHISHING = 1 (the "positive" / threat class).
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

RAW_TARGET = "label"
CLASS_NAMES = ["Legitimate", "Phishing"]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
TEXT_COLS = ["URL", "Domain", "TLD", "Title"]
# Scores the dataset authors precomputed against known-legit URL lists: they leak the label
# (one column alone gives MCC = 1.0), so they are excluded from every model.
LEAKY_COLS = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb", "CharContinuationRate"]
BAIT_WORDS = ("login", "signin", "verify", "secure", "account", "update", "confirm",
              "bank", "wallet", "password", "free", "bonus")
SKEWED = ["URLLength", "DomainLength", "LineOfCode", "LargestLineLength", "NoOfImage", "NoOfCSS",
          "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef", "NoOfLettersInURL",
          "NoOfDegitsInURL", "NoOfOtherSpecialCharsInURL"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def url_signals(urls: pd.Series) -> pd.DataFrame:
    u = urls.fillna("").astype(str)
    host = u.str.replace(r"^[a-zA-Z]+://", "", regex=True).str.split("/").str[0]
    path = u.str.replace(r"^[a-zA-Z]+://[^/]*", "", regex=True)
    low = u.str.lower()
    return pd.DataFrame({
        "sig_dots_in_host": host.str.count(r"\."),
        "sig_hyphens_in_host": host.str.count("-"),
        "sig_has_at": u.str.contains("@").astype(int),
        "sig_path_depth": path.str.count("/"),
        "sig_query_len": u.str.split("?", n=1).str[1].fillna("").str.len(),
        "sig_digits_in_host": host.str.count(r"\d"),
        "sig_host_entropy": host.map(_entropy),
        "sig_bait_words": low.map(lambda s: sum(w in s for w in BAIT_WORDS)),
    }, index=urls.index)


def make_features(X) -> pd.DataFrame:
    """Raw dataset columns -> model features. First step of every saved pipeline."""
    X = pd.DataFrame(X).copy()
    if "URL" in X.columns:
        X = pd.concat([X, url_signals(X["URL"])], axis=1)
    g = lambda c: X[c].astype(float) if c in X.columns else pd.Series(0.0, index=X.index)
    refs = g("NoOfSelfRef") + g("NoOfExternalRef") + g("NoOfEmptyRef")
    X["ext_ref_share"] = g("NoOfExternalRef") / (refs + 1)
    X["empty_ref_share"] = g("NoOfEmptyRef") / (refs + 1)
    X["code_density"] = g("LineOfCode") / (g("LargestLineLength") + 1)
    X["risky_form"] = ((g("HasPasswordField") == 1) &
                       ((g("IsHTTPS") == 0) | (g("HasExternalFormSubmit") == 1))).astype(int)
    X["hollow_page"] = ((g("HasTitle") == 0).astype(int) + (g("HasFavicon") == 0).astype(int) +
                        (g("HasDescription") == 0).astype(int) + (g("HasCopyrightInfo") == 0).astype(int))
    X["bait_without_https"] = (X.get("sig_bait_words", 0) > 0).astype(int) * (g("IsHTTPS") == 0).astype(int)
    for c in SKEWED:
        if c in X.columns:
            X[c] = np.log1p(X[c].astype(float).clip(lower=0))
    drop = [c for c in TEXT_COLS + LEAKY_COLS + [RAW_TARGET] if c in X.columns]
    return X.drop(columns=drop).astype(float).fillna(0.0)


def to_binary(label: pd.Series) -> np.ndarray:
    """raw label (1=legit, 0=phish) -> 1 = phishing."""
    return (pd.Series(label).astype(int) == 0).astype(int).to_numpy()


def evaluate(y, pred, p_phish) -> dict:
    m = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, p_phish) if len(set(y)) == 2 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return {k: m[k] for k in METRIC_ORDER}
