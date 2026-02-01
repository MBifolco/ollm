"""
Adaptive Early-Exit Analysis

Simulates confidence-threshold exit policies using per-example margins.
Produces AUC-latency Pareto curves for different threshold values.

Usage:
    python src/analyze_adaptive_exit.py \
        --results results/unified_early_exit/semantic_seed42_testR.json \
        --output results/adaptive_exit/semantic_seed42.json
"""
from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


def simulate_adaptive_exit(
    results_path: str,
    candidate_layers: Optional[list[int]] = None,
    tau_values: Optional[list[float]] = None,
    min_exit_layer: int = 14,
) -> dict:
    """
    Simulate adaptive early-exit with confidence thresholds.

    Policy:
        For each example, try layers in order.
        At layer L, compute confidence = max(sigmoid(margin), 1-sigmoid(margin)).
        If confidence >= tau, exit at this layer.
        Otherwise continue to next layer.
        If no layer exits, use the final layer.

    Args:
        results_path: Path to JSON with per_example margins
        candidate_layers: Layers to try (default: [14, 16, 18, 23])
        tau_values: Threshold values to sweep (default: 0.5 to 0.99)
        min_exit_layer: Never exit before this layer (default: 14)

    Returns:
        Dict with per-threshold results and summary
    """
    with open(results_path) as f:
        data = json.load(f)

    # Check for per-example data
    if "per_example" not in data:
        raise ValueError(
            f"No per_example data in {results_path}. "
            "Re-run early_exit.py with --save_per_example"
        )

    examples = data["per_example"]
    per_layer = data["per_layer"]
    metadata = data["metadata"]

    # Get available layers from per_example margins
    available_layers = sorted([int(k) for k in examples[0]["margins"].keys()])

    # Default candidate layers (respecting min_exit_layer)
    if candidate_layers is None:
        candidate_layers = [L for L in available_layers if L >= min_exit_layer]

    # Filter to available layers
    candidate_layers = [L for L in candidate_layers if L in available_layers]
    if not candidate_layers:
        raise ValueError(f"No valid candidate layers. Available: {available_layers}")

    # Default tau sweep
    if tau_values is None:
        tau_values = np.linspace(0.5, 0.99, 50).tolist()

    # Get latency per layer
    latency_by_layer = {int(k): v["mean_latency_ms"] for k, v in per_layer.items()}

    # Full model latency (for reference)
    full_latency = data["full_forward"]["mean_latency_ms"]
    full_auc = data["full_forward"]["auc"]

    print(f"Simulating adaptive exit with {len(examples)} examples")
    print(f"Candidate layers: {candidate_layers}")
    print(f"Tau sweep: {len(tau_values)} values from {tau_values[0]:.2f} to {tau_values[-1]:.2f}")
    print(f"Full forward: AUC={full_auc:.4f}, latency={full_latency:.2f}ms")

    results = {
        "metadata": {
            "source_file": results_path,
            "model_path": metadata["model_path"],
            "n_examples": len(examples),
            "candidate_layers": candidate_layers,
            "min_exit_layer": min_exit_layer,
            "full_auc": full_auc,
            "full_latency_ms": full_latency
        },
        "per_threshold": []
    }

    for tau in tau_values:
        exit_layers = []
        exit_margins = []
        labels = []

        for ex in examples:
            label = ex["label"]
            margins = ex["margins"]

            # Find exit layer
            chosen_layer = candidate_layers[-1]  # Default to last
            chosen_margin = float(margins[str(chosen_layer)])

            for L in candidate_layers:
                m = float(margins[str(L)])
                p = sigmoid(m)
                conf = max(p, 1 - p)

                if conf >= tau:
                    chosen_layer = L
                    chosen_margin = m
                    break

            exit_layers.append(chosen_layer)
            exit_margins.append(chosen_margin)
            labels.append(label)

        # Compute metrics
        try:
            auc = roc_auc_score(labels, exit_margins)
        except ValueError:
            auc = None

        # Accuracy at tau (using margin sign)
        preds = [1 if m > 0 else 0 for m in exit_margins]
        accuracy = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)

        # Latency
        mean_latency = np.mean([latency_by_layer[L] for L in exit_layers])
        speedup = full_latency / mean_latency if mean_latency > 0 else 0

        # Exit distribution
        exit_dist = {str(L): int(np.sum(np.array(exit_layers) == L)) for L in candidate_layers}
        mean_exit_layer = np.mean(exit_layers)
        pct_early = sum(1 for L in exit_layers if L < candidate_layers[-1]) / len(exit_layers) * 100

        results["per_threshold"].append({
            "tau": float(tau),
            "auc": float(auc) if auc else None,
            "accuracy": float(accuracy),
            "mean_latency_ms": float(mean_latency),
            "speedup": float(speedup),
            "mean_exit_layer": float(mean_exit_layer),
            "pct_early_exit": float(pct_early),
            "exit_distribution": exit_dist
        })

    # Find Pareto-optimal points (for summary)
    pareto_points = []
    best_auc_so_far = 0
    for r in sorted(results["per_threshold"], key=lambda x: x["mean_latency_ms"]):
        if r["auc"] and r["auc"] > best_auc_so_far:
            pareto_points.append({
                "tau": r["tau"],
                "auc": r["auc"],
                "speedup": r["speedup"],
                "mean_exit_layer": r["mean_exit_layer"]
            })
            best_auc_so_far = r["auc"]

    results["pareto_frontier"] = pareto_points

    # Key thresholds summary
    key_taus = [0.80, 0.90, 0.95]
    results["key_thresholds"] = []
    for target_tau in key_taus:
        # Find closest tau
        closest = min(results["per_threshold"], key=lambda x: abs(x["tau"] - target_tau))
        results["key_thresholds"].append({
            "tau": target_tau,
            "actual_tau": closest["tau"],
            "auc": closest["auc"],
            "accuracy": closest["accuracy"],
            "speedup": closest["speedup"],
            "mean_exit_layer": closest["mean_exit_layer"],
            "exit_distribution": closest["exit_distribution"]
        })

    return results


def print_summary(results: dict):
    """Print a summary of adaptive exit results."""
    meta = results["metadata"]
    print("\n" + "=" * 70)
    print("ADAPTIVE EARLY-EXIT SUMMARY")
    print("=" * 70)
    print(f"Model: {meta['model_path']}")
    print(f"Full forward: AUC={meta['full_auc']:.4f}, latency={meta['full_latency_ms']:.2f}ms")
    print(f"Candidate layers: {meta['candidate_layers']}")

    print("\n--- Key Thresholds ---")
    print(f"{'Tau':<8} {'AUC':<10} {'Accuracy':<10} {'Speedup':<10} {'Mean Layer':<12} {'Exit Dist'}")
    print("-" * 70)

    for kt in results["key_thresholds"]:
        dist_str = ", ".join(f"L{k}:{v}" for k, v in kt["exit_distribution"].items() if v > 0)
        auc_str = f"{kt['auc']:.4f}" if kt['auc'] else "N/A"
        print(f"{kt['tau']:<8.2f} {auc_str:<10} {kt['accuracy']:<10.2%} {kt['speedup']:<10.2f}x {kt['mean_exit_layer']:<12.1f} {dist_str}")

    print("\n--- Pareto Frontier (speedup vs AUC) ---")
    if results["pareto_frontier"]:
        for p in results["pareto_frontier"][:5]:  # Top 5
            print(f"  τ={p['tau']:.2f}: AUC={p['auc']:.4f}, speedup={p['speedup']:.2f}x, mean_layer={p['mean_exit_layer']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive early-exit")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to JSON with per_example margins")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for adaptive analysis results")
    parser.add_argument("--candidate_layers", type=str, default="14,16,18,23",
                       help="Comma-separated list of candidate exit layers")
    parser.add_argument("--min_exit_layer", type=int, default=14,
                       help="Never exit before this layer")
    args = parser.parse_args()

    candidate_layers = [int(x.strip()) for x in args.candidate_layers.split(",")]

    results = simulate_adaptive_exit(
        results_path=args.results,
        candidate_layers=candidate_layers,
        min_exit_layer=args.min_exit_layer
    )

    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
