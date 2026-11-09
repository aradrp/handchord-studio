#!/usr/bin/env python3
"""
Train the gesture classifier on the collected dataset.

Steps:
  1) Hold out entire recording sessions or participants before augmentation
  2) Augmentation on the training set only
  3) Train an SVM and a Random Forest and compare their validation scores
  4) Retrain the best model on all data and save it
"""
import argparse
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from hand_features import augment, features_from_normalized


def load_dataset(path, *, group_by=None):
    labels, norms, groups = [], [], []
    column = 65 if group_by == "participant" else 64
    with open(path, newline="") as f:
        for line, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) not in (64, 65, 66):
                raise ValueError(f"{path}:{line}: expected a label, 63 coordinates, and optional recording IDs")
            labels.append(row[0])
            norms.append(np.array(row[1:64], dtype=float).reshape(21, 3))
            groups.append(row[column].strip() if len(row) > column else "")
    if group_by == "participant" and not any(groups):
        raise ValueError("No participant IDs. Collect data with --participant before using participant validation.")
    dataset = norms, np.array(labels)
    return (*dataset, np.array(groups, dtype=str)) if group_by else dataset


def validation_rows(labels, groups):
    """Use only labeled recording groups when measuring generalization."""
    if groups is None or not np.any(groups != ""):
        return np.arange(len(labels)), None
    rows = np.flatnonzero(groups != "")
    missing = set(labels) - set(labels[rows])
    if missing:
        raise ValueError(f"Record new data with group IDs for these gestures: {', '.join(sorted(missing))}.")
    return rows, groups[rows]


def validation_split(labels, groups, test_size, seed):
    rows, groups = validation_rows(labels, groups)
    if groups is None:
        return train_test_split(rows, test_size=test_size, stratify=labels, random_state=seed)

    classes = set(labels)
    if any(len(set(groups[labels[rows] == label])) < 2 for label in classes):
        raise ValueError("Record every gesture in at least two groups (sessions or participants) for validation.")
    splitter = GroupShuffleSplit(n_splits=64, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(rows, groups=groups):
        train_rows, test_rows = rows[train_idx], rows[test_idx]
        if set(labels[train_rows]) == set(labels[test_rows]) == classes:
            return train_rows, test_rows
    raise ValueError("Could not split complete gesture sets by group. Record all gestures in each group or adjust --test-size.")


def build_features(norms, labels, n_aug, rng):
    X, y = [], []
    for pts, lab in zip(norms, labels):
        X.append(features_from_normalized(pts))
        y.append(lab)
        for _ in range(n_aug):
            X.append(features_from_normalized(augment(pts, rng)))
            y.append(lab)
    return np.array(X), np.array(y)


def candidate_models(seed, proba=True):
    """proba=False skips SVM probability calibration (much faster; fine for cross-validation)."""
    return {
        "svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=proba,
                                                   random_state=seed)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
    }


def train(norms, labels, n_aug=4, test_size=0.2, seed=42, verbose=True, *, groups=None):
    rng = np.random.default_rng(seed)
    tr, te = validation_split(labels, groups, test_size, seed)

    X_tr, y_tr = build_features([norms[i] for i in tr], labels[tr], n_aug, rng)
    X_te, y_te = build_features([norms[i] for i in te], labels[te], 0, rng)

    results = {}
    for name, model in candidate_models(seed).items():
        model.fit(X_tr, y_tr)
        results[name] = (model.score(X_te, y_te), model)
        if verbose:
            print(f"  {name:<14} validation accuracy = {results[name][0]:.3f}")

    best = max(results, key=lambda k: results[k][0])
    acc, best_model = results[best]
    pred = best_model.predict(X_te)
    if verbose:
        print(f"\nBest model: {best}\n")
        print(classification_report(y_te, pred, digits=3))

    classes = sorted(set(labels))
    cm = confusion_matrix(y_te, pred, labels=classes)

    X_all, y_all = build_features(norms, labels, n_aug, rng)
    final = clone(best_model).fit(X_all, y_all)
    return final, best, acc, classes, cm


def save_confusion(cm, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_PATH))
    ap.add_argument("--out", default=str(config.MODEL_PATH))
    ap.add_argument("--augment", type=int, default=4, help="augmented copies per sample")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--group-by", choices=("session", "participant"), default="session")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        norms, labels, groups = load_dataset(args.dataset, group_by=args.group_by)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    counts = Counter(labels)
    print("Samples:", {str(k): v for k, v in counts.items()})
    if len(counts) < 2:
        raise SystemExit("At least two gestures are required.")
    few = [k for k, v in counts.items() if v < 20]
    if few:
        raise SystemExit(f"Gestures {few} have fewer than 20 samples; record more.")
    missing = [g for g in config.GESTURE_DEGREES if g not in counts]
    if missing:
        print(f"[!] No data for these gestures, so they will not be recognized: {missing}")

    validation = f"{args.group_by} holdout" if np.any(groups != "") else "frame split"
    if validation == "frame split":
        print("[!] No session IDs. Frame-split accuracy does not measure generalization to new sessions.")
    print(f"\nTraining with {validation} (the SVM can take a little while)...")
    try:
        model, best, acc, classes, cm = train(
            norms, labels, args.augment, args.test_size, args.seed, groups=groups
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_), "best": best,
                 "accuracy": acc, "validation": validation, "version": 1}, args.out)
    save_confusion(cm, classes, config.MODEL_DIR / "confusion_matrix.png")
    print(f"Model saved: {args.out}")


if __name__ == "__main__":
    main()
