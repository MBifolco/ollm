#!/usr/bin/env python3
"""
K=4 "Support" Style Leakage Check

Tests whether a shallow TF-IDF classifier can predict K=4 labels from surface style.
For K=4, random chance is 25%. If TF-IDF exceeds ~40-50%, there may be leakage.

Usage:
    python src/style_leakage_k4_support.py --data data/k4_support/pilot.jsonl
    python src/style_leakage_k4_support.py --data data/k4_support/pilot_minimal_pairs.jsonl --group_field base_id
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

CATEGORIES = ["E", "P", "I", "S"]
CATEGORY_NAMES = {
    "E": "Emotional",
    "P": "Practical",
    "I": "Ideological",
    "S": "Structural",
}


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
    by_difficulty: bool = False,
    remove_stopwords: bool = False,
    group_field: str | None = None,
):
    """Run TF-IDF + LogisticRegression to check for style leakage."""

    data = load_jsonl(data_path)
    texts = [ex.get("scenario", "") for ex in data]
    labels = [ex.get("label", "UNK") for ex in data]
    groups = [ex.get(group_field) if group_field else None for ex in data]

    print(f"Loaded {len(data)} examples from {data_path}")
    label_counts = Counter(labels)
    print(f"Label distribution: {label_counts}")

    if by_difficulty:
        difficulties = [ex.get("difficulty", "unknown") for ex in data]
        diff_counts = Counter(difficulties)
        print(f"Difficulty distribution: {diff_counts}")

    # Vectorize - optionally remove stopwords to separate noun vs voice leakage
    stop_words_setting = "english" if remove_stopwords else None
    print(f"Stopwords: {'removed' if remove_stopwords else 'kept (includes pronouns)'}")

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

    def cross_validated_accuracy(X_cv, y_cv, groups_cv=None, group_field_name: str | None = None) -> tuple[np.ndarray, str]:
        """Return CV accuracy scores and a short description of the split strategy."""
        clf_cv = LogisticRegression(max_iter=1000, random_state=42)

        try:
            if groups_cv is not None:
                n_groups = len(set(groups_cv))
                n_splits = min(5, n_groups)
                if n_splits < 2:
                    raise ValueError(f"Need >=2 groups for GroupKFold, got {n_groups}")
                cv = GroupKFold(n_splits=n_splits)
                scores = cross_val_score(clf_cv, X_cv, y_cv, groups=groups_cv, cv=cv, scoring="accuracy")
                field_name = group_field_name or group_field
                return scores, f"{n_splits}-fold GroupKFold (group_field={field_name})"

            class_counts = Counter(y_cv.tolist())
            n_splits = min(5, min(class_counts.values()))
            if n_splits < 2:
                raise ValueError(f"Need >=2 samples per class for StratifiedKFold, got {class_counts}")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf_cv, X_cv, y_cv, cv=cv, scoring="accuracy")
            return scores, f"{n_splits}-fold StratifiedKFold"
        except ValueError as e:
            print(f"Cross-validation failed: {e}")
            return np.array([np.nan]), "CV failed"

    groups_arr = None
    auto_groups_arr = None
    if group_field:
        groups_arr = np.array(groups)
        n_groups = len(set(groups_arr.tolist()))
        print(f"Grouping enabled: field={group_field} (groups={n_groups})")
    else:
        # Heuristic warning: minimal-pairs datasets often include base_id and require grouped CV.
        if data and "base_id" in data[0]:
            base_ids = [ex.get("base_id") for ex in data]
            if any(b is None for b in base_ids):
                print("NOTE: Detected 'base_id' field but some rows have null base_id; skipping grouped leakage hint.")
            elif len(set(base_ids)) < len(base_ids):
                print("NOTE: Detected repeated 'base_id' groups (minimal pairs). Row-wise StratifiedKFold can be misleading.")
                print("      Re-run with: --group_field base_id  (recommended)")
                auto_groups_arr = np.array(base_ids)
            else:
                print("NOTE: Detected 'base_id' field but no repeats; treating as ungrouped data.")

    scores, cv_desc = cross_validated_accuracy(X, y, groups_arr)

    print(f"\n{'='*60}")
    print("TF-IDF STYLE LEAKAGE CHECK (K=4 Support)")
    print(f"{'='*60}")
    print(f"Random chance: 25%")
    print(f"Acceptable range: 25-40%")
    print(f"Concern threshold: >50%")
    if np.isnan(scores).any():
        print(f"\n{cv_desc}: unavailable")
        mean_score = float("nan")
    else:
        mean_score = scores.mean()
        print(f"\n{cv_desc} accuracy: {mean_score:.1%} (+/- {scores.std()*2:.1%})")

    # If minimal-pair grouping is present but not used, also print grouped CV as a sanity check.
    grouped_scores = None
    grouped_desc = None
    if groups_arr is None and auto_groups_arr is not None:
        grouped_scores, grouped_desc = cross_validated_accuracy(
            X, y, auto_groups_arr, group_field_name="base_id"
        )
        if grouped_scores is not None and not np.isnan(grouped_scores).any():
            print(f"\n{grouped_desc} accuracy (auto-check): {grouped_scores.mean():.1%} (+/- {grouped_scores.std()*2:.1%})")

    # Decide which metric to use for leakage warnings.
    # For minimal-pair datasets, grouped CV is the meaningful metric.
    assessed_score = mean_score
    assessed_desc = cv_desc
    if groups_arr is not None:
        assessed_score = mean_score
        assessed_desc = cv_desc
    elif grouped_scores is not None and not np.isnan(grouped_scores).any():
        assessed_score = grouped_scores.mean()
        assessed_desc = f"{grouped_desc} (preferred for minimal pairs)"

    if not np.isnan(assessed_score) and assessed_score > 0.50:
        print(f"\n⚠️  WARNING: TF-IDF accuracy {assessed_score:.1%} > 50% ({assessed_desc})")
        print("   Dataset likely has style shortcuts!")
    elif not np.isnan(assessed_score) and assessed_score > 0.40:
        print(f"\n⚠️  CAUTION: TF-IDF accuracy {assessed_score:.1%} is borderline ({assessed_desc})")
        print("   Inspect top features below.")
    elif not np.isnan(assessed_score):
        print(f"\n✓ TF-IDF accuracy {assessed_score:.1%} is acceptable ({assessed_desc})")
    else:
        print("\n⚠️  Could not compute CV accuracy (see error above).")

    # Fit full model to get feature importances
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    # Show top features per class
    if show_top_features > 0:
        print(f"\n{'='*60}")
        print(f"TOP {show_top_features} FEATURES PER CATEGORY (potential leakage)")
        print(f"{'='*60}")

        for i, cat in enumerate(clf.classes_):
            coefs = clf.coef_[i]
            top_idx = np.argsort(coefs)[-show_top_features:][::-1]
            top_features = [(feature_names[j], coefs[j]) for j in top_idx]

            cat_name = CATEGORY_NAMES.get(cat, cat)
            print(f"\n{cat} ({cat_name}):")
            for feat, score in top_features:
                print(f"  {feat:30} {score:+.3f}")

    # Check for verb overlap (good sign)
    print(f"\n{'='*60}")
    print("VERB OVERLAP CHECK")
    print(f"{'='*60}")

    common_verbs = ["stayed", "talked", "listened", "worked", "helped",
                    "discussed", "read", "built", "fixed", "organized"]
    verb_by_class = defaultdict(list)

    for i, cat in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        for verb in common_verbs:
            for j, feat in enumerate(feature_names):
                if verb in feat:
                    if coefs[j] > 0.1:
                        verb_by_class[cat].append((feat, coefs[j]))

    for cat in clf.classes_:
        cat_name = CATEGORY_NAMES.get(cat, cat)
        verbs = verb_by_class.get(cat, [])
        if verbs:
            print(f"\n{cat} ({cat_name}) strong verb features:")
            for feat, score in sorted(verbs, key=lambda x: -x[1])[:5]:
                print(f"  {feat:30} {score:+.3f}")
        else:
            print(f"\n{cat} ({cat_name}): no dominant verb features (good)")

    # Difficulty breakdown if requested
    if by_difficulty and "difficulty" in data[0]:
        print(f"\n{'='*60}")
        print("ACCURACY BY DIFFICULTY")
        print(f"{'='*60}")

        for diff in ["easy", "hard"]:
            diff_idx = [i for i, ex in enumerate(data) if ex.get("difficulty") == diff]
            if not diff_idx:
                continue
            X_diff = X[diff_idx]
            y_diff = y[diff_idx]
            groups_diff = None
            if groups_arr is not None:
                groups_diff = groups_arr[diff_idx]
            elif auto_groups_arr is not None:
                groups_diff = auto_groups_arr[diff_idx]

            scores_diff, cv_desc_diff = cross_validated_accuracy(X_diff, y_diff, groups_diff)
            if np.isnan(scores_diff).any():
                print(f"\n{diff.upper()}: {cv_desc_diff} unavailable ({len(diff_idx)} examples)")
            else:
                print(f"\n{diff.upper()}: {cv_desc_diff} accuracy {scores_diff.mean():.1%} ({len(diff_idx)} examples)")

    return clf, vectorizer, assessed_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K=4 Support style leakage check")
    parser.add_argument("--data", type=str, required=True, help="JSONL data file")
    parser.add_argument("--top_features", type=int, default=10, help="Show top N features per class")
    parser.add_argument("--by_difficulty", action="store_true", help="Break down by difficulty")
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="Remove English stopwords (to isolate noun/template leakage from voice leakage)")
    parser.add_argument("--group_field", type=str, default=None,
                        help="Use GroupKFold grouped by this JSON field (recommended for minimal-pair datasets)")
    args = parser.parse_args()

    run_leakage_check(
        data_path=args.data,
        show_top_features=args.top_features,
        by_difficulty=args.by_difficulty,
        remove_stopwords=args.remove_stopwords,
        group_field=args.group_field,
    )
