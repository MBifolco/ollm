"""
Experiment 3A: Style leakage sanity check.

Tests whether a shallow TF-IDF classifier can predict labels from surface style alone.
If it can, the dataset has style shortcuts. Rewrites should break these shortcuts.

Usage:
    python src/exp3a_style_leakage.py --train data/train.jsonl --test data/test.jsonl
    python src/exp3a_style_leakage.py --train data/train.jsonl --test data/test_rewritten.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np


@dataclass
class StyleLeakageResults:
    """Results from style leakage check."""
    train_file: str
    test_file: str
    accuracy: float
    auc: float
    romantic_precision: float
    romantic_recall: float
    nonromantic_precision: float
    nonromantic_recall: float


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_style_leakage_check(
    train_path: str,
    test_path: str,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    first_n_tokens: int = None,
    verbose: bool = True
) -> StyleLeakageResults:
    """
    Run TF-IDF + LogisticRegression to check for style leakage.

    Args:
        train_path: Path to training JSONL
        test_path: Path to test JSONL
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        first_n_tokens: If set, only use first N tokens of each scenario
        verbose: Print detailed results

    Returns:
        StyleLeakageResults with metrics
    """
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    def extract_text(example: dict) -> str:
        text = example.get("scenario", "")
        if first_n_tokens:
            tokens = text.split()[:first_n_tokens]
            text = " ".join(tokens)
        return text

    def extract_label(example: dict) -> int:
        return 1 if example.get("label") == "romantic" else 0

    train_texts = [extract_text(ex) for ex in train_data]
    train_labels = [extract_label(ex) for ex in train_data]
    test_texts = [extract_text(ex) for ex in test_data]
    test_labels = [extract_label(ex) for ex in test_data]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # Train logistic regression
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=['non-romantic', 'romantic'], output_dict=True)

    results = StyleLeakageResults(
        train_file=train_path,
        test_file=test_path,
        accuracy=accuracy,
        auc=auc,
        romantic_precision=report['romantic']['precision'],
        romantic_recall=report['romantic']['recall'],
        nonromantic_precision=report['non-romantic']['precision'],
        nonromantic_recall=report['non-romantic']['recall']
    )

    if verbose:
        print("\n" + "="*70)
        print("STYLE LEAKAGE CHECK (TF-IDF + Logistic Regression)")
        print("="*70)
        print(f"\nTrain: {train_path} ({len(train_data)} examples)")
        print(f"Test:  {test_path} ({len(test_data)} examples)")
        if first_n_tokens:
            print(f"Using first {first_n_tokens} tokens only")
        print(f"\nVectorizer: ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}")
        print(f"Features: {X_train.shape[1]}")

        print(f"\n{'Metric':<25} {'Value':>10}")
        print("-"*40)
        print(f"{'Accuracy':<25} {accuracy:>10.2%}")
        print(f"{'AUC':<25} {auc:>10.4f}")
        print(f"{'Romantic Precision':<25} {results.romantic_precision:>10.2%}")
        print(f"{'Romantic Recall':<25} {results.romantic_recall:>10.2%}")
        print(f"{'Non-romantic Precision':<25} {results.nonromantic_precision:>10.2%}")
        print(f"{'Non-romantic Recall':<25} {results.nonromantic_recall:>10.2%}")

        # Interpretation
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        if accuracy > 0.70:
            print(f"⚠️  High accuracy ({accuracy:.1%}) suggests significant style leakage.")
            print("   A shallow classifier can exploit surface patterns to predict labels.")
        elif accuracy > 0.55:
            print(f"⚡ Moderate accuracy ({accuracy:.1%}) suggests some style leakage.")
            print("   Some surface patterns correlate with labels.")
        else:
            print(f"✓  Low accuracy ({accuracy:.1%}) suggests minimal style leakage.")
            print("   Surface style doesn't strongly predict labels.")

        # Show top features if leakage is high
        if accuracy > 0.55:
            feature_names = vectorizer.get_feature_names_out()
            coefs = clf.coef_[0]

            # Top romantic-leaning features
            top_rom_idx = np.argsort(coefs)[-10:][::-1]
            print("\nTop romantic-leaning features:")
            for idx in top_rom_idx:
                print(f"  {feature_names[idx]:<30} {coefs[idx]:>+.3f}")

            # Top non-romantic-leaning features
            top_nonrom_idx = np.argsort(coefs)[:10]
            print("\nTop non-romantic-leaning features:")
            for idx in top_nonrom_idx:
                print(f"  {feature_names[idx]:<30} {coefs[idx]:>+.3f}")

    return results


def compare_original_vs_rewritten(
    train_path: str,
    test_original_path: str,
    test_rewritten_path: str
) -> dict:
    """
    Compare style leakage on original vs rewritten test sets.

    This is the key comparison for Experiment 3A:
    - If TF-IDF drops significantly on rewrites, the rewrites broke style shortcuts
    - If token model drops less than TF-IDF, the token learned semantics not style
    """
    print("\n" + "="*70)
    print("COMPARING ORIGINAL vs REWRITTEN TEST SETS")
    print("="*70)

    print("\n>>> Original test set:")
    original_results = run_style_leakage_check(train_path, test_original_path, verbose=True)

    print("\n>>> Rewritten test set:")
    rewritten_results = run_style_leakage_check(train_path, test_rewritten_path, verbose=True)

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    acc_delta = rewritten_results.accuracy - original_results.accuracy
    auc_delta = rewritten_results.auc - original_results.auc

    print(f"\n{'Metric':<20} {'Original':>12} {'Rewritten':>12} {'Delta':>12}")
    print("-"*60)
    print(f"{'Accuracy':<20} {original_results.accuracy:>12.2%} {rewritten_results.accuracy:>12.2%} {acc_delta:>+12.2%}")
    print(f"{'AUC':<20} {original_results.auc:>12.4f} {rewritten_results.auc:>12.4f} {auc_delta:>+12.4f}")

    if acc_delta < -0.10:
        print("\n✓ TF-IDF accuracy dropped significantly on rewrites.")
        print("  This suggests the rewrites successfully broke style shortcuts.")
    elif acc_delta < -0.05:
        print("\n⚡ TF-IDF accuracy dropped moderately on rewrites.")
        print("  Some style shortcuts were broken, but some remain.")
    else:
        print("\n⚠️  TF-IDF accuracy did not drop much on rewrites.")
        print("  The rewrites may not have changed surface style enough.")

    return {
        "original": original_results,
        "rewritten": rewritten_results,
        "accuracy_delta": acc_delta,
        "auc_delta": auc_delta
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style leakage check via TF-IDF")
    parser.add_argument("--train", type=str, default="data/train.jsonl", help="Training data path")
    parser.add_argument("--test", type=str, default="data/test.jsonl", help="Test data path")
    parser.add_argument("--test_rewritten", type=str, default=None,
                        help="Rewritten test data path (for comparison mode)")
    parser.add_argument("--first_n_tokens", type=int, default=None,
                        help="Only use first N tokens (for quick leak test)")
    parser.add_argument("--ngram_min", type=int, default=1, help="Min n-gram size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Max n-gram size")
    args = parser.parse_args()

    if args.test_rewritten:
        # Comparison mode
        compare_original_vs_rewritten(args.train, args.test, args.test_rewritten)
    else:
        # Single test mode
        run_style_leakage_check(
            args.train,
            args.test,
            ngram_range=(args.ngram_min, args.ngram_max),
            first_n_tokens=args.first_n_tokens
        )
