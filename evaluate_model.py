#!/usr/bin/env python3
"""
Scientific evaluation of the gesture classifier - useful numbers for a report.

  1) Grouped k-fold cross-validation of SVM / Random Forest / k-NN
     when recording IDs exist, with augmentation applied to the training folds only
  2) Per-class recall and a pooled confusion matrix (models/cv_confusion_matrix.png)
  3) Per-frame latency benchmark of the classification step
     (features + predict_proba for a single hand)

Usage:  python evaluate_model.py [--folds 5] [--augment 2]
"""
import argparse
import time
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from hand_features import features_from_normalized
from train_model import build_features, candidate_models, load_dataset, save_confusion, validation_rows


def models_for_cv(seed, *, proba=False):
    m = candidate_models(seed, proba=proba)
    m["knn"] = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    return m


def validation_folds(labels, groups, k, seed):
    rows, groups = validation_rows(labels, groups)
    classes = set(labels)
    if k < 2 or len(classes) < 2:
        raise ValueError("Cross-validation needs at least two folds and two gestures.")
    if groups is None:
        if min(Counter(labels).values()) < k:
            raise ValueError(f"Each gesture needs at least {k} samples for {k}-fold CV.")
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    else:
        if any(len(set(groups[labels[rows] == label])) < k for label in classes):
            raise ValueError(f"Each gesture needs at least {k} groups (sessions or participants) for {k}-fold CV.")
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(rows[tr], rows[te]) for tr, te in splitter.split(rows, labels[rows], groups)]
    if any(set(labels[tr]) != classes or set(labels[te]) != classes for tr, te in folds):
        raise ValueError("Some folds lack gestures. Record all gestures in each group or use fewer folds.")
    return folds


def cross_validate(norms, labels, k=5, n_aug=2, seed=42, *, groups=None):
    """Returns {model: {"acc": [per-fold], "y_true": [...], "y_pred": [...]}}."""
    rng = np.random.default_rng(seed)
    results = {n: {"acc": [], "y_true": [], "y_pred": []} for n in models_for_cv(seed)}
    for tr, te in validation_folds(labels, groups, k, seed):
        X_tr, y_tr = build_features([norms[i] for i in tr], labels[tr], n_aug, rng)
        X_te, y_te = build_features([norms[i] for i in te], labels[te], 0, rng)
        for name, model in models_for_cv(seed).items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            results[name]["acc"].append(float(np.mean(pred == y_te)))
            results[name]["y_true"].extend(y_te)
            results[name]["y_pred"].extend(pred)
    return results


def latency_benchmark(model, norms, n=200):
    """Average ms per hand for features + predict_proba (single-sample calls, like the live app)."""
    sample = [norms[i % len(norms)] for i in range(n)]
    model.predict_proba(features_from_normalized(sample[0])[None, :])      # warm-up
    t0 = time.perf_counter()
    for pts in sample:
        model.predict_proba(features_from_normalized(pts)[None, :])
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATASET_PATH))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--augment", type=int, default=2)
    ap.add_argument("--group-by", choices=("session", "participant"), default="session")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        norms, labels, groups = load_dataset(args.dataset, group_by=args.group_by)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    counts = Counter(labels)
    print("Samples:", {str(k): v for k, v in counts.items()})
    validation = f"{args.group_by}-grouped" if np.any(groups != "") else "frame-level"
    if validation == "frame-level":
        print("[!] No session IDs. Frame-level CV does not measure generalization to new sessions.")
    print(f"\n{args.folds}-fold {validation} cross-validation (augment x{args.augment}):")
    try:
        res = cross_validate(norms, labels, args.folds, args.augment, args.seed, groups=groups)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"  {'model':<14}{'mean acc':>10}{'std':>8}")
    for name, r in res.items():
        print(f"  {name:<14}{np.mean(r['acc']):>10.3f}{np.std(r['acc']):>8.3f}")
    best = max(res, key=lambda n: np.mean(res[n]["acc"]))

    classes = sorted(set(labels))
    cm = confusion_matrix(res[best]["y_true"], res[best]["y_pred"], labels=classes)
    print(f"\nPer-class recall ({best}):")
    for i, c in enumerate(classes):
        print(f"  {c:<12}{cm[i, i] / max(cm[i].sum(), 1):>7.3f}")
    config.MODEL_DIR.mkdir(exist_ok=True)
    save_confusion(cm, classes, config.MODEL_DIR / "cv_confusion_matrix.png")

    X, y = build_features(norms, labels, 0, np.random.default_rng(args.seed))
    final = models_for_cv(args.seed, proba=True)[best].fit(X, y)
    ms = latency_benchmark(final, norms)
    print(f"\nClassification latency ({best}): {ms:.3f} ms per hand  ->  up to {1000 / ms:.0f} classifications/s")
    print("(MediaPipe landmark detection is extra; read the FPS shown in app.py for the full pipeline.)")


if __name__ == "__main__":
    main()
