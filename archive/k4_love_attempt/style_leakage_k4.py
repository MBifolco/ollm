"""
K=4 Style leakage sanity check.

Tests whether a shallow TF-IDF classifier can predict K=4 labels from surface style.
For K=4, random chance is 25%. If TF-IDF exceeds ~35-40%, there's leakage.

Usage:
    python src/style_leakage_k4.py --data data_k4/stream.jsonl
    python src/style_leakage_k4.py --train data_k4/train.jsonl --test data_k4/test.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

CATEGORIES = ["ROM", "FAM", "PLA", "OBJ"]


def load_jsonl(path: str) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_leakage_check(
    data_path: str = None,
    train_path: str = None,
    test_path: str = None,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    show_top_features: int = 10,
):
    """Run TF-IDF + LogisticRegression to check for style leakage."""

    # Load data
    if data_path:
        # Single file: use cross-validation
        data = load_jsonl(data_path)
        texts = [ex.get("scenario", "") for ex in data]
        labels = [ex.get("label", "UNK") for ex in data]

        print(f"Loaded {len(data)} examples from {data_path}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # Vectorize
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words=None  # Keep pronouns - "you/this/that" are discriminative!
        )
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Cross-validation
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        print(f"\n{'='*60}")
        print("TF-IDF STYLE LEAKAGE CHECK (K=4)")
        print(f"{'='*60}")
        print(f"Random chance: 25%")
        print(f"Leakage threshold: ~35-40%")
        print(f"\n5-fold CV accuracy: {scores.mean():.1%} (+/- {scores.std()*2:.1%})")

        if scores.mean() > 0.40:
            print(f"\n⚠️  WARNING: TF-IDF accuracy {scores.mean():.1%} > 40%")
            print("   Dataset likely has style shortcuts!")
        elif scores.mean() > 0.35:
            print(f"\n⚠️  CAUTION: TF-IDF accuracy {scores.mean():.1%} is borderline")
            print("   Inspect top features below.")
        else:
            print(f"\n✓ TF-IDF accuracy {scores.mean():.1%} is acceptable")

        # Fit full model to get feature importances
        clf.fit(X, y)

    else:
        # Train/test split provided
        train_data = load_jsonl(train_path)
        test_data = load_jsonl(test_path)

        train_texts = [ex.get("scenario", "") for ex in train_data]
        train_labels = [ex.get("label", "UNK") for ex in train_data]
        test_texts = [ex.get("scenario", "") for ex in test_data]
        test_labels = [ex.get("label", "UNK") for ex in test_data]

        print(f"Train: {len(train_data)} examples")
        print(f"Test: {len(test_data)} examples")

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words=None  # Keep pronouns - "you/this/that" are discriminative!
        )

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*60}")
        print("TF-IDF STYLE LEAKAGE CHECK (K=4)")
        print(f"{'='*60}")
        print(f"Random chance: 25%")
        print(f"Leakage threshold: ~35-40%")
        print(f"\nTest accuracy: {acc:.1%}")

        if acc > 0.40:
            print(f"\n⚠️  WARNING: TF-IDF accuracy {acc:.1%} > 40%")
            print("   Dataset likely has style shortcuts!")
        elif acc > 0.35:
            print(f"\n⚠️  CAUTION: TF-IDF accuracy {acc:.1%} is borderline")
        else:
            print(f"\n✓ TF-IDF accuracy {acc:.1%} is acceptable")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=CATEGORIES))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
        print(f"{'':>8}", end="")
        for cat in CATEGORIES:
            print(f"{cat:>8}", end="")
        print()
        for i, cat in enumerate(CATEGORIES):
            print(f"{cat:>8}", end="")
            for j in range(len(CATEGORIES)):
                print(f"{cm[i,j]:>8}", end="")
            print()

    # Show top features per class
    if show_top_features > 0:
        feature_names = vectorizer.get_feature_names_out()
        print(f"\n{'='*60}")
        print(f"TOP {show_top_features} FEATURES PER CATEGORY (potential leakage)")
        print(f"{'='*60}")

        for i, cat in enumerate(clf.classes_):
            coefs = clf.coef_[i]
            top_idx = np.argsort(coefs)[-show_top_features:][::-1]
            top_features = [(feature_names[j], coefs[j]) for j in top_idx]

            print(f"\n{cat}:")
            for feat, score in top_features:
                print(f"  {feat:30} {score:+.3f}")

    return clf, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K=4 style leakage check")
    parser.add_argument("--data", type=str, help="Single JSONL file (uses cross-validation)")
    parser.add_argument("--train", type=str, help="Training JSONL file")
    parser.add_argument("--test", type=str, help="Test JSONL file")
    parser.add_argument("--top_features", type=int, default=10, help="Show top N features per class")
    args = parser.parse_args()

    if args.data:
        run_leakage_check(data_path=args.data, show_top_features=args.top_features)
    elif args.train and args.test:
        run_leakage_check(train_path=args.train, test_path=args.test, show_top_features=args.top_features)
    else:
        parser.error("Provide either --data or both --train and --test")
