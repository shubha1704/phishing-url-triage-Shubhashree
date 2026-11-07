# 🎣 Phish Triage Desk - Phishing Website Classification

## a. Problem statement
Phishing pages imitate trusted sites to steal credentials. Security teams must decide, for every page,
whether to block it - and the two mistakes are not equally expensive (a missed phishing page costs far more
than a wrongly blocked safe page). This project trains five classifiers to label a web page as
**Phishing** or **Legitimate** from its URL structure and page content signals, and wraps them in a
Streamlit "triage desk" where an analyst picks a model, sets a blocking threshold, and sees the resulting
confusion matrix and cost. Models are loaded from disk; nothing is retrained on uploaded data.

## b. Dataset description
- **Source:** PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, id 967; also on Kaggle).
- **Size:** 235,795 web pages x 54 raw columns; no missing values, no duplicate rows.
- **Target:** `label` (raw dataset convention: 1 = legitimate, 0 = phishing). Class balance is 57.2 % legitimate / 42.8 % phishing. Internally the code treats **phishing as the positive class**.
- **Raw features:** URL length, domain length, subdomain count, obfuscation, HTTPS, special-character ratios, page statistics (lines of code, images, CSS/JS files, references), and flags such as favicon, password field, hidden fields, external form submit, bank/pay/crypto keywords.
- **Leakage handling:** four precomputed score columns (`URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, `CharContinuationRate`) are excluded. In my check, `URLSimilarityIndex` alone carried ~99 % of a depth-3 tree's importance and a tree using it reached MCC = 1.0, because it is computed against known-legitimate pages.
- **Own feature engineering** (`phish_core.py`): URL signals parsed from the raw URL (dots/hyphens in host, path depth, query length, digits in host, host entropy, bait words such as *login/verify/secure*), reference shares (external/empty), code density, a risky-form flag (password field on a non-HTTPS page or external submit), a hollow-page score and bait-word-without-HTTPS; count features are log-transformed.
- **Split:** 10,000 stratified pages are held out as `test_data.csv`; the other 225,795 are used for training. 3-fold `GridSearchCV` (scoring = MCC) is run on the training portion only. SVM and KNN are trained on a stratified 30,000-row sample for speed; the other three use all training rows.

## c. GitHub repository link
https://github.com/shubha1704/phishing-url-triage-Shubhashree

Live app: https://phishing-url-triage-sp.streamlit.app/

```
app.py            Streamlit app
phish_core.py     feature engineering, metrics (shared by train.py and app.py)
train.py          tuning, evaluation, model export
requirements.txt
test_data.csv     10,000 held-out pages (raw columns + label)
model/            5 saved pipelines (.joblib) + meta.json
```

## d. Models used
Metrics are on the 10,000-row `test_data.csv` with a 0.5 threshold; Precision, Recall and F1 are for the Phishing class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9999 | 1.0000 | 1.0000 | 0.9998 | 0.9999 | 0.9998 |
| Decision Tree | 0.9998 | 0.9998 | 1.0000 | 0.9995 | 0.9998 | 0.9996 |
| KNN | 0.9986 | 0.9998 | 1.0000 | 0.9967 | 0.9984 | 0.9971 |
| Naive Bayes | 0.9989 | 0.9994 | 0.9986 | 0.9988 | 0.9987 | 0.9978 |
| Support Vector Machine | 0.9997 | 1.0000 | 0.9993 | 1.0000 | 0.9996 | 0.9994 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This is the best model. It scored the highest overall (0.9998) and missed almost no phishing pages. The data is easy to separate, so a simple model works very well. |
| Decision Tree | Almost as good (0.9996). It never marked a safe page as phishing, but it missed some of the phishing pages. It works by asking simple yes or no questions, which suits this data. |
| KNN | The weakest of the five (0.9971). It missed the most phishing pages, about 14 out of roughly 4,280. It is slow on big data, so I trained it on 30,000 rows, and that likely held it back. |
| Naive Bayes | A little behind the best models (0.9978). It treats every feature as unrelated to the others, but things like URL length and digit count are related, so it loses a little accuracy. |
| Support Vector Machine | Very strong (0.9994). It caught every phishing page, but it wrongly blocked a few safe pages. I trained it on 30,000 rows. |
| **Overall winner** | Logistic Regression, because it has the best overall score. All five models are very close, so the difference is tiny. If catching every phishing page matters most, the Support Vector Machine is the safest choice. |

## How to run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```
