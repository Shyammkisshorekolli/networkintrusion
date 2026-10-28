from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Keeping paths relative to this file lets the script run from VS Code or a terminal.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "DATA"
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
DATASETS = {"cic": ("CIC-UNSW-NB15", 40, 0.30),
            "hikari": ("HIKARI-2021", 50, 0.30),
            "unr": ("UNR-IDD", 20, 0.20)}
MODEL_VERSION = "paper-method-v4-no-leakage"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_dump(value: Any, path: Path) -> None:
    # Write to a temporary file first so an interruption won't leave a half-written cache.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    dump(value, temp, compress=3)
    os.replace(temp, path)


def fingerprint(paths: list[Path], settings: dict[str, Any]) -> str:
    # Reuse work only when both the source files and the run settings still match.
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def load_manifest() -> dict[str, Any]:
    path = OUTPUTS / "run_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"runs": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "run_manifest.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp, path)


def dataset_paths(name: str) -> list[Path]:
    if name == "cic": return [DATA / "CIC-UNSW-NB15" / "data.csv", DATA / "CIC-UNSW-NB15" / "Label.csv"]
    if name == "hikari": return [DATA / "HIKARI-2021" / "ALLFLOWMETER_HIKARI2021.csv"]
    full_file = DATA / "UNR-IDD.csv"
    if full_file.exists(): return [full_file]
    return [DATA / "UNR_IDD" / f"{part}.csv" for part in ("train", "val", "test")]


def read_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, list[Path]]:
    """Read all available rows; every dataset is split below using the paper's ratio."""
    if name == "cic":
        x_path = DATA / "CIC-UNSW-NB15" / "data.csv"
        y_path = DATA / "CIC-UNSW-NB15" / "Label.csv"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Expected {x_path} and {y_path}")
        x = pd.read_csv(x_path, low_memory=False)
        raw_labels = pd.read_csv(y_path, header=None, low_memory=False)
        if raw_labels.shape[1] != 1:
            raise ValueError(f"CIC Label.csv must have exactly one column; found {raw_labels.shape[1]}")
        y = raw_labels.iloc[:, 0]
        # Accept either a true header or a headerless label file without shifting labels.
        if len(y) == len(x) + 1 and str(y.iloc[0]).strip().lower() == "label":
            y = y.iloc[1:].reset_index(drop=True)
        if len(x) != len(y):
            raise ValueError(f"CIC rows do not align: {len(x):,} features vs {len(y):,} labels")
        return x, y, [x_path, y_path]
    if name == "hikari":
        path = DATA / "HIKARI-2021" / "ALLFLOWMETER_HIKARI2021.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        columns = {str(c).strip().lower(): c for c in frame.columns}
        target = columns.get("traffic_category")
        if target is None:
            raise ValueError("HIKARI CSV has no traffic_category attack-label column")
        # Label is the binary normal/attack flag; exclude it and use attack type as target.
        features = frame.drop(columns=[target, columns["label"]] if "label" in columns else [target])
        return features, frame[target], [path]

    full_file = DATA / "UNR-IDD.csv"
    if full_file.exists():
        frame = pd.read_csv(full_file, low_memory=False)
        target = next((c for c in frame.columns if str(c).strip().lower() == "label"), None)
        if target is None:
            raise ValueError(f"No multiclass Label column in {full_file}")
        return frame.drop(columns=target), frame[target], [full_file]

    paths = [DATA / "UNR_IDD" / f"{part}.csv" for part in ("train", "val", "test")]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"Expected DATA/UNR-IDD.csv or train/val/test files under {DATA / 'UNR_IDD'}")
    features, labels = [], []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        target = next((c for c in frame.columns if str(c).strip().lower() == "label"), None)
        if target is None:
            raise ValueError(f"No Label column in {path}")
        labels.append(frame.pop(target))
        features.append(frame)
    # Reassemble all provided partitions before applying the paper's 80/20 split.
    return pd.concat(features, ignore_index=True), pd.concat(labels, ignore_index=True), paths


def clean_dataset(x: pd.DataFrame, y: pd.Series, name: str) -> tuple[pd.DataFrame, pd.Series]:
    x = x.copy()
    columns = {str(c).strip().lower(): c for c in x.columns}
    excluded: set[Any] = set()
    if name == "hikari":
        excluded.update(c for c in x.columns if str(c).strip().lower().startswith("unnamed"))
        excluded.update(columns[c] for c in ("uid", "originh", "originp", "responh", "responp") if c in columns)
    if name == "unr":
        # These network/device identifiers were removed in the paper's real-world setup.
        excluded.update(columns[c] for c in ("switch id", "port number", "binary label") if c in columns)
        excluded.update(c for c in x.columns if str(c).strip().lower().startswith("unnamed"))
    x = x.drop(columns=list(excluded), errors="ignore")
    x.columns = [str(c).strip() for c in x.columns]
    x = x.loc[:, ~x.columns.duplicated()].reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True).astype("string").fillna("<missing>")
    if len(x) != len(y):
        raise ValueError("Feature and label counts differ")
    combined = x.copy()
    combined["__target__"] = y
    if name in ("cic", "hikari"):
        before = len(combined)
        combined = combined.drop_duplicates().reset_index(drop=True)
        log(f"Removed {before - len(combined):,} duplicate feature/label rows")
    return combined.drop(columns="__target__"), combined["__target__"]


def encode_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_train = x_train.copy()
    x_test = x_test.copy().reindex(columns=x_train.columns)
    categorical = [c for c in x_train if not pd.api.types.is_numeric_dtype(x_train[c])]
    numeric = [c for c in x_train if c not in categorical]
    for column in numeric:
        # Learn fill values from training rows only; the test set must not influence preprocessing.
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(x_train[column].median()) if x_train[column].notna().any() else 0.0
        x_train[column] = x_train[column].fillna(median).astype("float32")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(median).astype("float32")
    if categorical:
        # Give unseen test categories a safe code rather than failing during transformation.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-1, dtype=np.float32)
        x_train[categorical] = encoder.fit_transform(x_train[categorical].astype("string").fillna("<missing>"))
        x_test[categorical] = encoder.transform(x_test[categorical].astype("string").fillna("<missing>"))
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    return x_train.to_numpy(dtype=np.float32), x_test.to_numpy(dtype=np.float32), list(x_train.columns)


def prepare(name: str, seed: int) -> dict[str, Any]:
    x, y, _ = read_dataset(name)
    x, y = clean_dataset(x, y, name)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    _, feature_count, test_size = DATASETS[name]
    # Keep every class represented in both partitions. Feature encoders below use training rows only.
    train_x, test_x, train_y, test_y = train_test_split(
        x, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded, shuffle=True)
    x_train, x_test, columns = encode_features(train_x, test_x)
    return {"x_train": x_train, "y_train": train_y, "x_test": x_test,
            "y_test": test_y, "columns": columns,
            "classes": [str(c) for c in target_encoder.classes_],
            "feature_count": feature_count, "test_size": test_size}


def select_features(data: dict[str, Any], destination: Path, force: bool, seed: int) -> tuple[list[str], list[int]]:
    if destination.exists() and not force:
        result = load(destination)
        log(f"Reusing RF-RFE selection: {len(result['indices'])} features")
        return result["names"], result["indices"]
    x, y = data["x_train"], data["y_train"]
    # RF-RFE repeatedly removes the least useful features until the paper's count remains.
    # It sees every training row, but never the held-out test partition.
    selector = RFE(
        estimator=RandomForestClassifier(n_estimators=100, max_features="sqrt", n_jobs=-1,
                                         random_state=seed),
        n_features_to_select=min(data["feature_count"], x.shape[1]), step=0.1)
    log(f"Running RF-RFE on all {len(y):,} training rows; this is the slow feature-selection stage")
    selector.fit(x, y)
    indices = np.flatnonzero(selector.support_).tolist()
    names = [data["columns"][i] for i in indices]
    atomic_dump({"names": names, "indices": indices}, destination)
    log(f"Saved selected features ({len(indices)})")
    return names, indices


def make_model(name: str, trial: Any, seed: int) -> Any:
    # Optuna supplies a trial here; refit_model uses the same factory with fixed best values.
    if name == "decision_tree":
        depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30, 50])
        return DecisionTreeClassifier(max_depth=depth,
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            n_jobs=-1, random_state=seed)
    if name == "naive_bayes":
        return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True))
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_categorical("n_neighbors", [3, 5, 7, 11, 15]),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2), n_jobs=-1)
    if name == "bagging":
        # The paper does not name Bagging's base learner. Its text links Bagging's
        # inference cost to distance-based methods, so use KNN as the documented inference.
        base_knn = KNeighborsClassifier(
            n_neighbors=trial.suggest_categorical("n_neighbors", [3, 5, 7]), n_jobs=1)
        return BaggingClassifier(
            estimator=base_knn,
            n_estimators=trial.suggest_categorical("n_estimators", [10, 25, 50, 100]),
            max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            n_jobs=-1, random_state=seed)
    if name == "svm":
        # Match SVC's standard RBF kernel; linear SVC on unscaled byte counters can be intractable.
        return SVC(C=trial.suggest_float("C", 0.1, 100, log=True),
                   gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                   cache_size=2000)
    if name == "mlp":
        layer_choice = trial.suggest_categorical("hidden_layer_sizes", ["64", "100", "100_50", "128_64"])
        layers = {"64": (64,), "100": (100,), "100_50": (100, 50), "128_64": (128, 64)}[layer_choice]
        return MLPClassifier(hidden_layer_sizes=layers,
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            max_iter=300, early_stopping=True, random_state=seed)
    if name == "boosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 250),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), random_state=seed)
    raise KeyError(name)


def build_stacking(seed: int) -> Any:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    # The paper names these three boosters but not its final learner; use logistic regression.
    return StackingClassifier(
        estimators=[
            ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                   eval_metric="mlogloss", random_state=seed)),
            ("lgbm", LGBMClassifier(n_estimators=200, verbosity=-1, n_jobs=-1, random_state=seed)),
            ("catboost", CatBoostClassifier(iterations=200, depth=6, verbose=False,
                                             thread_count=-1, random_seed=seed)),
        ], final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=1)


def tuned_models(dataset: str) -> list[str]:
    # The paper explicitly omits SVM, MLP and standalone gradient boosting on the large files.
    names = ["decision_tree", "random_forest", "naive_bayes", "knn", "bagging"]
    if dataset == "unr":
        names += ["svm", "mlp", "boosting"]
    return names


def tune(name: str, x: np.ndarray, y: np.ndarray, database: Path,
         trials: int, folds: int, seed: int) -> tuple[dict[str, Any], float]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    database.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{database.resolve().as_posix()}"
    # Loading the same SQLite study lets a stopped run continue its unfinished trials.
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True,
                                direction="maximize")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    def objective(trial: Any) -> float:
        estimator = make_model(name, trial, seed)
        # Score each candidate on the training partition only; the test set is saved for later.
        scores = cross_val_score(estimator, x, y, scoring="accuracy", cv=cv, n_jobs=1,
                                 error_score="raise")
        return float(scores.mean())
    completed = sum(t.state.name == "COMPLETE" for t in study.trials)
    remaining = max(0, trials - completed)
    if remaining:
        log(f"Optuna {name}: {completed}/{trials} completed; running {remaining} trials × {folds}-fold CV")
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    else:
        log(f"Reusing completed Optuna study for {name} ({completed} trials)")
    if study.best_trial is None:
        raise RuntimeError(f"Optuna has no completed trials for {name}")
    return dict(study.best_params), float(study.best_value)


def refit_model(name: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    class FixedTrial:
        def suggest_categorical(self, key: str, choices: list[Any]) -> Any: return params[key]
        def suggest_int(self, key: str, low: int, high: int, **kwargs: Any) -> int: return int(params[key])
        def suggest_float(self, key: str, low: float, high: float, **kwargs: Any) -> float: return float(params[key])
    return make_model(name, FixedTrial(), seed).fit(x, y)


def evaluate(name: str, model: Any, x: np.ndarray, y: np.ndarray,
             labels: list[str], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    predicted = model.predict(x)
    inference_seconds = time.perf_counter() - started
    # These are weighted multiclass metrics; inference time covers predict() on the test rows.
    precision = float(precision_score(y, predicted, average="weighted", zero_division=0))
    recall = float(recall_score(y, predicted, average="weighted", zero_division=0))
    report = classification_report(y, predicted, labels=np.arange(len(labels)),
                                   target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=np.arange(len(labels)))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")
    (output / f"{name}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return {"model": name, "accuracy": float(accuracy_score(y, predicted)),
            "precision": precision, "recall": recall,
            "f1_score": float(f1_score(y, predicted, average="weighted", zero_division=0)),
            "missed_alarm_rate": 1.0 - recall, "false_alarm_rate": 1.0 - precision,
            "inference_time_seconds": inference_seconds, "test_rows": int(len(y))}


def run_dataset(name: str, args: argparse.Namespace) -> None:
    title, feature_count, test_size = DATASETS[name]
    settings = {"dataset": name, "seed": args.seed, "cv_folds": args.cv_folds,
                "trials": args.trials, "features": feature_count,
                "test_size": test_size, "skip_rfe": args.skip_rfe,
                "model_version": MODEL_VERSION}
    source_paths = dataset_paths(name)
    if not all(path.exists() for path in source_paths):
        raise FileNotFoundError(f"Missing one or more input files for {title} under {DATA}")
    # The fingerprint gives each data/settings combination its own resumable cache folder.
    run_id = fingerprint(source_paths, settings)
    if args.force:
        run_id = f"{run_id}-fresh-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cache = CACHE / name / run_id
    output = OUTPUTS / name / run_id
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    entry = manifest["runs"].setdefault(run_id, {"dataset": title, "completed": [],
        "started": datetime.now(timezone.utc).isoformat()})
    entry.update(status="running", settings=settings)
    save_manifest(manifest)

    # Preprocessing and selected features are expensive on these large files, so save them once.
    prepared_path = cache / "prepared.joblib"
    if prepared_path.exists() and not args.force:
        data = load(prepared_path)
        log(f"{title}: reusing saved preprocessing")
    else:
        log(f"{title}: loading, cleaning, shuffling and making {100*(1-test_size):.0f}/{100*test_size:.0f} split")
        data = prepare(name, args.seed)
        atomic_dump(data, prepared_path)
        entry["completed"].append("preprocessing")
        entry["rows"] = {"train": len(data["y_train"]), "test": len(data["y_test"])}
        entry["input_features"] = len(data["columns"])
        save_manifest(manifest)
        log(f"Saved prepared data ({len(data['y_train']):,} train / {len(data['y_test']):,} test rows)")

    if args.skip_rfe:
        feature_names = data["columns"]
        feature_indices = list(range(len(feature_names)))
    else:
        feature_names, feature_indices = select_features(
            data, cache / "selected_features.joblib", args.force, args.seed)
    x_train = data["x_train"][:, feature_indices]
    x_test = data["x_test"][:, feature_indices]
    (output / "selected_features.txt").write_text("\n".join(feature_names), encoding="utf-8")
    if "feature_selection" not in entry["completed"]:
        entry["completed"].append("feature_selection")
        save_manifest(manifest)

    metrics_path = output / "metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() and not args.force else []
    complete = {str(row["model"]) for row in rows}
    for model_name in tuned_models(name):
        # Keep the new inferred KNN-Bagging artifacts separate from earlier tree-Bagging runs.
        artifact_name = "bagging_knn" if model_name == "bagging" else model_name
        model_path = cache / f"model_{artifact_name}.joblib"
        study_path = cache / f"optuna_{artifact_name}.sqlite3"
        if model_name in complete and model_path.exists() and not args.force:
            log(f"{title}: reusing finished {model_name}")
            continue
        # Each model is tuned and saved separately, so later runs can skip completed work.
        best_params, cv_accuracy = tune(model_name, x_train, data["y_train"], study_path,
                                        args.trials, args.cv_folds, args.seed)
        if model_path.exists() and not args.force:
            model = load(model_path)
            training_seconds = None
            log(f"{title}: reusing fitted {model_name}")
        else:
            started = time.perf_counter()
            model = refit_model(model_name, best_params, x_train, data["y_train"], args.seed)
            training_seconds = time.perf_counter() - started
            atomic_dump(model, model_path)
            log(f"{title}: saved best {model_name}; CV accuracy={cv_accuracy:.4f}")
        row = evaluate(model_name, model, x_test, data["y_test"], data["classes"], output)
        row.update(best_cv_accuracy=cv_accuracy, best_params=json.dumps(best_params),
                   training_seconds=training_seconds)
        rows = [r for r in rows if r["model"] != model_name] + [row]
        pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        complete.add(model_name)
        entry["completed_models"] = sorted(complete)
        save_manifest(manifest)

    if not args.skip_stacking:
        model_name = "stacking"
        model_path = cache / "model_stacking.joblib"
        if not (model_name in complete and model_path.exists() and not args.force):
            if model_path.exists() and not args.force:
                model = load(model_path)
                training_seconds = None
                log(f"{title}: reusing fitted stacking model")
            else:
                # The paper reports this stack without the Optuna search used for other models.
                log(f"{title}: fitting untuned XGBoost + LightGBM + CatBoost stacking model")
                started = time.perf_counter()
                model = build_stacking(args.seed).fit(x_train, data["y_train"])
                training_seconds = time.perf_counter() - started
                atomic_dump(model, model_path)
            row = evaluate(model_name, model, x_test, data["y_test"], data["classes"], output)
            row.update(best_cv_accuracy=None, best_params="paper: no extensive stacking tuning",
                       training_seconds=training_seconds)
            rows = [r for r in rows if r["model"] != model_name] + [row]
            pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
            complete.add(model_name)
            entry["completed_models"] = sorted(complete)
            save_manifest(manifest)

    entry.update(status="complete", finished=datetime.now(timezone.utc).isoformat(),
                 result_file=str(metrics_path))
    save_manifest(manifest)
    log(f"Completed {title}; results: {metrics_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-method NIDS reproduction with resumable checkpoints")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--trials", type=int, default=20,
                        help="Optuna trials per tuned model; paper does not state its trial count")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Stratified CV folds; paper does not state its fold count")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducibility seed; paper does not report its seed")
    parser.add_argument("--force", action="store_true", help="Start a fresh run/cache namespace")
    parser.add_argument("--skip-rfe", action="store_true", help="Bypass RFE and use every feature")
    parser.add_argument("--skip-stacking", action="store_true", help="Skip the untuned boosting stack")
    args = parser.parse_args()
    if args.trials < 1 or args.cv_folds < 2:
        parser.error("--trials must be positive and --cv-folds must be at least 2")
    CACHE.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    failures = 0
    for name in (list(DATASETS) if args.dataset == "all" else [args.dataset]):
        try:
            run_dataset(name, args)
        except Exception as exc:
            failures += 1
            log(f"ERROR {name}: {exc}")
            traceback.print_exc()
            manifest = load_manifest()
            for entry in manifest.get("runs", {}).values():
                if entry.get("status") == "running" and entry.get("dataset") == DATASETS[name][0]:
                    entry.update(status="error", error=str(exc),
                                 finished=datetime.now(timezone.utc).isoformat())
            save_manifest(manifest)
    try:
        # Keep the presentation file in sync after the requested dataset run(s) finish.
        from make_report import generate_report
        report_path = generate_report()
        log(f"Results report updated: {report_path}")
    except Exception as exc:
        log(f"WARNING: could not update results report: {exc}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

