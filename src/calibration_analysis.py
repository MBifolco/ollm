"""
Calibration Analysis for Early-Exit Models

Computes reliability curves, ECE, and Brier scores per layer to verify
that semantic init produces calibrated confidence for adaptive exit.

Usage:
    python src/calibration_analysis.py \
        --results results/adaptive_eval/semantic_seed42_per_example.json \
        --output results/calibration/semantic_seed42_calibration.json
"""
from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


def compute_calibration_metrics(
    margins: list[float],
    labels: list[int],
    n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics for a set of margins and labels.

    Args:
        margins: List of margin values (rom_logit - nonrom_logit)
        labels: List of ground truth labels (1=romantic, 0=non-romantic)
        n_bins: Number of bins for reliability curve

    Returns:
        Dict with ECE, Brier score, reliability curve data, accuracy vs confidence
    """
    n = len(margins)
    if n == 0:
        return {"error": "no examples"}

    # Compute probabilities and confidence
    probs = [sigmoid(m) for m in margins]
    confs = [max(p, 1 - p) for p in probs]
    preds = [1 if p >= 0.5 else 0 for p in probs]

    # Accuracy
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / n

    # Brier score (for prob of class 1)
    brier = sum((p - l) ** 2 for p, l in zip(probs, labels)) / n

    # Binning for reliability curve and ECE
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)  # Confidence is in [0.5, 1.0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include right edge for last bin
        if i == n_bins - 1:
            mask = [(low <= c <= high) for c in confs]
        else:
            mask = [(low <= c < high) for c in confs]

        bin_labels = [l for l, m in zip(labels, mask) if m]
        bin_preds = [p for p, m in zip(preds, mask) if m]
        bin_conf_vals = [c for c, m in zip(confs, mask) if m]

        count = len(bin_labels)
        bin_counts.append(count)

        if count > 0:
            bin_acc = sum(1 for p, l in zip(bin_preds, bin_labels) if p == l) / count
            bin_conf = sum(bin_conf_vals) / count
        else:
            bin_acc = None
            bin_conf = None

        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)

    # ECE: weighted average of |acc - conf| per bin
    ece = 0.0
    total_weight = 0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        if acc is not None and conf is not None and count > 0:
            ece += count * abs(acc - conf)
            total_weight += count
    ece = ece / total_weight if total_weight > 0 else 0.0

    # MCE: maximum calibration error
    mce = 0.0
    for acc, conf in zip(bin_accs, bin_confs):
        if acc is not None and conf is not None:
            mce = max(mce, abs(acc - conf))

    # Accuracy vs confidence deciles (for monotonicity check)
    decile_edges = np.linspace(0.5, 1.0, 11)
    decile_accs = []
    for i in range(10):
        low, high = decile_edges[i], decile_edges[i + 1]
        if i == 9:
            mask = [(low <= c <= high) for c in confs]
        else:
            mask = [(low <= c < high) for c in confs]

        decile_labels = [l for l, m in zip(labels, mask) if m]
        decile_preds = [p for p, m in zip(preds, mask) if m]
        count = len(decile_labels)

        if count > 0:
            dec_acc = sum(1 for p, l in zip(decile_preds, decile_labels) if p == l) / count
            decile_accs.append({
                "decile": i + 1,
                "conf_range": f"{low:.2f}-{high:.2f}",
                "count": count,
                "accuracy": dec_acc
            })
        else:
            decile_accs.append({
                "decile": i + 1,
                "conf_range": f"{low:.2f}-{high:.2f}",
                "count": 0,
                "accuracy": None
            })

    # Check monotonicity: is accuracy increasing with confidence?
    valid_accs = [d["accuracy"] for d in decile_accs if d["accuracy"] is not None]
    is_monotonic = all(valid_accs[i] <= valid_accs[i + 1] for i in range(len(valid_accs) - 1)) if len(valid_accs) > 1 else True

    # Mean confidence by true label (sanity check)
    mean_conf_pos = np.mean([c for c, l in zip(confs, labels) if l == 1]) if sum(labels) > 0 else None
    mean_conf_neg = np.mean([c for c, l in zip(confs, labels) if l == 0]) if sum(l == 0 for l in labels) > 0 else None

    # Mean margin by true label (sign check)
    mean_margin_pos = np.mean([m for m, l in zip(margins, labels) if l == 1]) if sum(labels) > 0 else None
    mean_margin_neg = np.mean([m for m, l in zip(margins, labels) if l == 0]) if sum(l == 0 for l in labels) > 0 else None

    return {
        "n_examples": n,
        "accuracy": accuracy,
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "is_monotonic": is_monotonic,
        "reliability_curve": {
            "bin_centers": bin_centers.tolist(),
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts
        },
        "accuracy_by_decile": decile_accs,
        "mean_conf_positive": mean_conf_pos,
        "mean_conf_negative": mean_conf_neg,
        "mean_margin_positive": mean_margin_pos,
        "mean_margin_negative": mean_margin_neg
    }


def analyze_calibration(
    results_path: str,
    layers: Optional[list[int]] = None,
    n_bins: int = 10
) -> dict:
    """
    Analyze calibration for a model across multiple layers.

    Args:
        results_path: Path to JSON with per_example margins
        layers: Layers to analyze (default: all available)
        n_bins: Number of bins for reliability curve

    Returns:
        Dict with per-layer calibration metrics
    """
    with open(results_path) as f:
        data = json.load(f)

    if "per_example" not in data:
        raise ValueError(f"No per_example data in {results_path}")

    examples = data["per_example"]
    metadata = data["metadata"]

    # Get available layers
    available_layers = sorted([int(k) for k in examples[0]["margins"].keys()])
    if layers is None:
        layers = available_layers
    else:
        layers = [L for L in layers if L in available_layers]

    print(f"Analyzing calibration for {len(examples)} examples")
    print(f"Layers: {layers}")

    results = {
        "metadata": {
            "source_file": results_path,
            "model_path": metadata.get("model_path", "unknown"),
            "n_examples": len(examples),
            "layers_analyzed": layers
        },
        "per_layer": {}
    }

    labels = [ex["label"] for ex in examples]

    for layer in layers:
        margins = [float(ex["margins"][str(layer)]) for ex in examples]
        metrics = compute_calibration_metrics(margins, labels, n_bins)
        results["per_layer"][str(layer)] = metrics

        print(f"\nLayer {layer}:")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Brier: {metrics['brier_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Monotonic: {metrics['is_monotonic']}")
        print(f"  Mean margin (pos/neg): {metrics['mean_margin_positive']:.2f} / {metrics['mean_margin_negative']:.2f}")

    return results


def print_summary(results: dict):
    """Print a summary table of calibration metrics."""
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Model: {results['metadata']['model_path']}")
    print(f"N examples: {results['metadata']['n_examples']}")

    print(f"\n{'Layer':<8} {'ECE':<10} {'Brier':<10} {'Accuracy':<10} {'Monotonic':<10}")
    print("-" * 50)

    for layer in sorted(results["per_layer"].keys(), key=int):
        m = results["per_layer"][layer]
        mono_str = "Yes" if m["is_monotonic"] else "No"
        print(f"{layer:<8} {m['ece']:<10.4f} {m['brier_score']:<10.4f} {m['accuracy']:<10.2%} {mono_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis for early-exit")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to JSON with per_example margins")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for calibration analysis")
    parser.add_argument("--layers", type=str, default="14,16,18,23",
                       help="Comma-separated list of layers to analyze")
    parser.add_argument("--n_bins", type=int, default=10,
                       help="Number of bins for reliability curve")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",")]

    results = analyze_calibration(
        results_path=args.results,
        layers=layers,
        n_bins=args.n_bins
    )

    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
