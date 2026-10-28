# Network Intrusion Detection System — paper method implementation

Open this folder (`network_intrusion`) in VS Code. Datasets are organized under `DATA/`. UNR uses `DATA/UNR-IDD.csv` when present; otherwise the loader falls back to `DATA/UNR_IDD/train.csv`, `val.csv` and `test.csv`.

## Setup and run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_nids.py --dataset unr --trials 5 --cv-folds 3
python run_nids.py --dataset all
```

In VS Code, Run and Debug also has **NIDS: UNR quick run** and **NIDS: all datasets**.

## What the script follows from the paper

- Uses CIC-UNSW-NB15, HIKARI-2021 and UNR-IDD separately.
- Removes duplicate feature/label rows from CIC and HIKARI (not UNR), then makes a stratified random split. HIKARI uses `traffic_category` as the attack-type target and removes binary `Label` from its inputs. UNR removes `Switch ID`, `Port Number` and `Binary Label` when present; `Connection Point` is retained.
- Uses 70/30 train/test splits for CIC and HIKARI and 80/20 for UNR. The full UNR CSV is split 80/20; if only the old three partitions are present, they are recombined before the 80/20 split.
- Applies RF-based recursive feature elimination using the complete training partition to select 40 CIC, 50 HIKARI or 20 UNR features.
- Tunes Decision Tree, Random Forest, Naive Bayes, KNN and Bagging with Optuna, maximizing accuracy using stratified cross-validation. On UNR it also tunes SVM, MLP and Gradient Boosting. The paper excludes those three models for the two larger datasets.
- Fits the XGBoost + LightGBM + CatBoost stacking ensemble without extensive hyperparameter tuning. The paper does not specify the meta-learner; this implementation uses Logistic Regression and serial base-estimator fitting to limit nested CPU/RAM use.
- Reports accuracy, weighted precision/recall/F1, missed-alarm and false-alarm rates, and total inference time on the held-out test set.

The paper does not state the random seed, number of CV folds, Optuna trial count, all search ranges, Bagging base learner, or stacking meta-learner. It links Bagging inference cost with distance-based models but does not name its base learner; this implementation uses KNN for Bagging as an explicit inference, and RBF SVM and Logistic Regression stacking meta-learner as reproducibility choices. Defaults here are seed 42, 5 stratified folds and 20 trials; these are explicit reproducibility choices, not values claimed by the paper. Adjust them with `--seed`, `--cv-folds` and `--trials`. Exact published scores are not guaranteed when those experimental details are unavailable.

## Resume and outputs

Preprocessed arrays, complete RF-RFE selections, fitted models, and Optuna trial histories are saved under `cache/`. Optuna uses a SQLite study for each model, so completed trials survive interruption and the next run continues the remaining trials. Metrics and per-model classification reports are saved under `outputs/` after each model finishes. Rerun the same command to resume. `--force` creates a fresh cache namespace without deleting previous results. `--skip-rfe` and `--skip-stacking` are optional runtime shortcuts and should stay off for the full paper workflow.

Full-data RF-RFE and 5-fold tuning can take hours and need substantial RAM, especially for CIC and HIKARI. Start with UNR to confirm the environment. Input CSV files are never changed.
## Presentation report

After a pipeline run, open `outputs/research_results_report.html` in a browser. It has one tab per dataset, paper-versus-project plots for accuracy and weighted F1, and a detailed model table with precision, recall, MAR, FAR, inference time and percentage-point differences. The pipeline refreshes the report when it finishes. To rebuild it later without rerunning model training:

```powershell
python make_report.py
```

Use the browser’s Print command and choose Save as PDF if you need a shareable copy. The report labels the run settings so quick runs are not mistaken for the full final run.
