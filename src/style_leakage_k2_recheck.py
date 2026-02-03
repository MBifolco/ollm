#!/usr/bin/env python3
"""
K=2 Style Leakage Recheck

Re-examines K=2 datasets with the methodology learned from K=4:
- Cross-validation (not just train/test split)
- GroupKFold by bucket (to test if bucket structure matters)
- With/without stopwords (to separate voice vs noun leakage)

Usage:
    python src/style_leakage_k2_recheck.py --data data/k2_love/O/train.jsonl
    python src/style_leakage_k2_recheck.py --data data/k2_love/O/train.jsonl --group_field bucket
    python src/style_leakage_k2_recheck.py --all  # Run on O, R, M
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_leakage_check(
    data_path: str,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    show_top_features: int = 10,
    remove_stopwords: bool = False,
    group_field: str | None = None,
) -> dict:
    """Run TF-IDF + LogisticRegression CV to check for style leakage."""

    data = load_jsonl(data_path)
    texts = [ex.get("scenario", "") for ex in data]

    # K=2 uses "romantic" / "non-romantic" labels
    labels = [ex.get("label", "UNK") for ex in data]
    groups = [ex.get(group_field) if group_field else None for ex in data]

    print(f"Loaded {len(data)} examples from {data_path}")
    label_counts = Counter(labels)
    print(f"Label distribution: {label_counts}")

    # Show bucket distribution if available
    if "bucket" in data[0]:
        bucket_counts = Counter(ex.get("bucket") for ex in data)
        print(f"Bucket distribution: {bucket_counts}")

    # Vectorize
    stop_words_setting = "english" if remove_stopwords else None
    print(f"Stopwords: {'removed' if remove_stopwords else 'kept'}")

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        stop_words=stop_words_setting
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    feature_names = vectorizer.get_feature_names_out()

    def cross_validated_accuracy(X_cv, y_cv, groups_cv=None) -> tuple[np.ndarray, str]:
        """Return CV accuracy scores and description."""
        clf_cv = LogisticRegression(max_iter=1000, random_state=42)

        try:
            if groups_cv is not None:
                n_groups = len(set(groups_cv))
                n_splits = min(5, n_groups)
                if n_splits < 2:
                    raise ValueError(f"Need >=2 groups for GroupKFold, got {n_groups}")
                cv = GroupKFold(n_splits=n_splits)
                scores = cross_val_score(clf_cv, X_cv, y_cv, groups=groups_cv, cv=cv, scoring="accuracy")
                return scores, f"{n_splits}-fold GroupKFold (group_field={group_field})"

            class_counts = Counter(y_cv.tolist())
            n_splits = min(5, min(class_counts.values()))
            if n_splits < 2:
                raise ValueError(f"Need >=2 samples per class for StratifiedKFold")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf_cv, X_cv, y_cv, cv=cv, scoring="accuracy")
            return scores, f"{n_splits}-fold StratifiedKFold"
        except ValueError as e:
            print(f"Cross-validation failed: {e}")
            return np.array([np.nan]), "CV failed"

    groups_arr = None
    if group_field:
        groups_arr = np.array(groups)
        n_groups = len(set(groups_arr.tolist()))
        print(f"Grouping enabled: field={group_field} (groups={n_groups})")

    scores, cv_desc = cross_validated_accuracy(X, y, groups_arr)

    print(f"\n{'='*60}")
    print("TF-IDF STYLE LEAKAGE CHECK (K=2)")
    print(f"{'='*60}")
    print(f"Random chance: 50%")
    print(f"Acceptable range: 50-60%")
    print(f"Concern threshold: >70%")

    if np.isnan(scores).any():
        print(f"\n{cv_desc}: unavailable")
        mean_score = float("nan")
    else:
        mean_score = scores.mean()
        print(f"\n{cv_desc} accuracy: {mean_score:.1%} (+/- {scores.std()*2:.1%})")

    if not np.isnan(mean_score) and mean_score > 0.70:
        print(f"\n⚠️  WARNING: TF-IDF accuracy {mean_score:.1%} > 70%")
        print("   Dataset likely has style shortcuts!")
    elif not np.isnan(mean_score) and mean_score > 0.60:
        print(f"\n⚠️  CAUTION: TF-IDF accuracy {mean_score:.1%} is borderline")
    elif not np.isnan(mean_score):
        print(f"\n✓ TF-IDF accuracy {mean_score:.1%} is acceptable")

    # Fit full model to get feature importances
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    # Show top features per class
    if show_top_features > 0:
        print(f"\n{'='*60}")
        print(f"TOP {show_top_features} FEATURES PER CLASS")
        print(f"{'='*60}")

        classes = clf.classes_
        coefs = clf.coef_[0]  # Binary classification has shape (1, n_features)

        # Top features for class 1 (positive coef)
        top_pos_idx = np.argsort(coefs)[-show_top_features:][::-1]
        print(f"\n{classes[1]} (positive coefficients):")
        for idx in top_pos_idx:
            print(f"  {feature_names[idx]:30} {coefs[idx]:+.3f}")

        # Top features for class 0 (negative coef)
        top_neg_idx = np.argsort(coefs)[:show_top_features]
        print(f"\n{classes[0]} (negative coefficients):")
        for idx in top_neg_idx:
            print(f"  {feature_names[idx]:30} {coefs[idx]:+.3f}")

    return {
        "data_path": data_path,
        "cv_desc": cv_desc,
        "accuracy": mean_score,
        "std": scores.std() if not np.isnan(scores).any() else np.nan,
        "n_examples": len(data),
        "stopwords_removed": remove_stopwords,
        "group_field": group_field,
    }


def run_all_datasets():
    """Run leakage check on all K=2 datasets."""
    datasets = [
        ("data/k2_love/O/train.jsonl", "Original (O)"),
        ("data/k2_love/R/train.jsonl", "Rewritten (R)"),
        ("data/k2_love/M/train.jsonl", "Mixed (M)"),
    ]

    results = []

    for data_path, name in datasets:
        if not Path(data_path).exists():
            print(f"Skipping {name}: {data_path} not found")
            continue

        print(f"\n{'#'*70}")
        print(f"# {name}: {data_path}")
        print(f"{'#'*70}")

        # StratifiedKFold, with stopwords
        print("\n--- StratifiedKFold, stopwords kept ---")
        r1 = run_leakage_check(data_path, remove_stopwords=False, show_top_features=5)
        r1["name"] = name
        r1["mode"] = "stratified+stopwords"
        results.append(r1)

        # StratifiedKFold, without stopwords
        print("\n--- StratifiedKFold, stopwords removed ---")
        r2 = run_leakage_check(data_path, remove_stopwords=True, show_top_features=5)
        r2["name"] = name
        r2["mode"] = "stratified-stopwords"
        results.append(r2)

        # GroupKFold by bucket, with stopwords
        print("\n--- GroupKFold by bucket, stopwords kept ---")
        r3 = run_leakage_check(data_path, group_field="bucket", remove_stopwords=False, show_top_features=5)
        r3["name"] = name
        r3["mode"] = "grouped+stopwords"
        results.append(r3)

        # GroupKFold by bucket, without stopwords
        print("\n--- GroupKFold by bucket, stopwords removed ---")
        r4 = run_leakage_check(data_path, group_field="bucket", remove_stopwords=True, show_top_features=5)
        r4["name"] = name
        r4["mode"] = "grouped-stopwords"
        results.append(r4)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'CV Mode':<25} {'Accuracy':>10} {'Std':>8}")
    print("-" * 60)

    for r in results:
        acc_str = f"{r['accuracy']:.1%}" if not np.isnan(r['accuracy']) else "N/A"
        std_str = f"±{r['std']*2:.1%}" if not np.isnan(r.get('std', np.nan)) else ""
        print(f"{r['name']:<12} {r['mode']:<25} {acc_str:>10} {std_str:>8}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K=2 style leakage recheck")
    parser.add_argument("--data", type=str, help="JSONL data file")
    parser.add_argument("--all", action="store_true", help="Run on all K=2 datasets (O, R, M)")
    parser.add_argument("--group_field", type=str, default=None,
                        help="Use GroupKFold grouped by this field (e.g., 'bucket')")
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="Remove English stopwords")
    parser.add_argument("--top_features", type=int, default=10,
                        help="Show top N features per class")
    args = parser.parse_args()

    if args.all:
        run_all_datasets()
    elif args.data:
        run_leakage_check(
            data_path=args.data,
            show_top_features=args.top_features,
            remove_stopwords=args.remove_stopwords,
            group_field=args.group_field,
        )
    else:
        print("Usage: --data <file> or --all")
