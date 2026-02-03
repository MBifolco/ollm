"""
Early-Exit Simulation (A1)

Computes the "potential compute saved" by analyzing at which layer
the model's decision is already good enough.

This is a SIMULATION - no actual early-exit is implemented.
It shows the ceiling for potential savings.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Optional


def load_layerwise_results(path: str = "layerwise_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def compute_early_exit_metrics(data: dict) -> dict:
    """Compute AUC and accuracy at each exit layer."""
    labels = data["labels"]
    n_layers = data["metadata"]["n_layers"]

    results = {"baseline": {}, "token": {}}

    for model_name in ["baseline", "token"]:
        results[model_name] = {
            "layers": [],
            "auc": [],
            "accuracy": [],
            "auc_ratio": [],  # AUC at layer / final AUC
        }

        # Get final layer AUC for ratio calculation
        final_margins = data["raw_margins"][model_name][str(n_layers)]
        try:
            final_auc = roc_auc_score(labels, final_margins)
        except ValueError:
            final_auc = None

        for layer_idx in range(n_layers + 1):
            margins = data["raw_margins"][model_name][str(layer_idx)]

            # Accuracy: margin > 0 means romantic (label=1)
            preds = [1 if m > 0 else 0 for m in margins]
            acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)

            # AUC
            try:
                auc = roc_auc_score(labels, margins)
            except ValueError:
                auc = None

            # AUC ratio
            if auc is not None and final_auc is not None and final_auc > 0:
                auc_ratio = auc / final_auc
            else:
                auc_ratio = None

            results[model_name]["layers"].append(layer_idx)
            results[model_name]["auc"].append(auc)
            results[model_name]["accuracy"].append(acc)
            results[model_name]["auc_ratio"].append(auc_ratio)

        results[model_name]["final_auc"] = final_auc

    return results


def find_earliest_threshold(aucs: list, threshold: float) -> Optional[int]:
    """Find earliest layer achieving AUC >= threshold."""
    for i, auc in enumerate(aucs):
        if auc is not None and auc >= threshold:
            return i
    return None


def compute_potential_savings(exit_layer: int, total_layers: int) -> float:
    """Compute potential compute saved (as fraction)."""
    return 1 - (exit_layer / total_layers)


def main():
    print("="*80)
    print("EARLY-EXIT SIMULATION (A1)")
    print("="*80)
    print("\nNOTE: This shows POTENTIAL savings. Actual FLOPs savings")
    print("      requires implementing early-exit (future work).\n")

    data = load_layerwise_results()
    results = compute_early_exit_metrics(data)

    n_layers = data["metadata"]["n_layers"]

    # Print AUC vs layer table
    print("\n" + "-"*80)
    print("AUC vs Exit Layer")
    print("-"*80)
    print(f"{'Layer':<8} {'Baseline AUC':<15} {'Token AUC':<15} {'Token AUC %':<15} {'Depth %':<10}")
    print("-"*80)

    for i in range(n_layers + 1):
        b_auc = results["baseline"]["auc"][i]
        t_auc = results["token"]["auc"][i]
        t_ratio = results["token"]["auc_ratio"][i]
        depth_pct = (i / n_layers) * 100

        b_str = f"{b_auc:.4f}" if b_auc else "N/A"
        t_str = f"{t_auc:.4f}" if t_auc else "N/A"
        r_str = f"{t_ratio*100:.1f}%" if t_ratio else "N/A"

        print(f"{i:<8} {b_str:<15} {t_str:<15} {r_str:<15} {depth_pct:.0f}%")

    # Summary statistics
    print("\n" + "="*80)
    print("EARLY-EXIT POTENTIAL SUMMARY")
    print("="*80)

    # Thresholds to check
    thresholds = [0.90, 0.95, 0.98]

    print(f"\n{'Threshold':<12} {'Baseline Layer':<18} {'Token Layer':<15} {'Token Depth %':<15} {'Potential Savings':<18}")
    print("-"*80)

    for thresh in thresholds:
        b_layer = find_earliest_threshold(results["baseline"]["auc"], thresh)
        t_layer = find_earliest_threshold(results["token"]["auc"], thresh)

        b_str = str(b_layer) if b_layer is not None else "never"
        t_str = str(t_layer) if t_layer is not None else "never"

        if t_layer is not None:
            depth_pct = (t_layer / n_layers) * 100
            savings = compute_potential_savings(t_layer, n_layers) * 100
            depth_str = f"{depth_pct:.0f}%"
            savings_str = f"{savings:.0f}%"
        else:
            depth_str = "N/A"
            savings_str = "N/A"

        print(f"AUC≥{thresh:<7} {b_str:<18} {t_str:<15} {depth_str:<15} {savings_str:<18}")

    # Final AUC comparison
    print(f"\n{'Final AUC':<12} {results['baseline']['final_auc']:.4f}{'':14} {results['token']['final_auc']:.4f}")

    # Key headline
    print("\n" + "="*80)
    print("KEY HEADLINE")
    print("="*80)

    t_95_layer = find_earliest_threshold(results["token"]["auc"], 0.95)
    b_95_layer = find_earliest_threshold(results["baseline"]["auc"], 0.95)

    if t_95_layer is not None:
        t_depth = (t_95_layer / n_layers) * 100
        t_savings = compute_potential_savings(t_95_layer, n_layers) * 100
        print(f"\n✓ Token model reaches AUC≥0.95 at layer {t_95_layer}/{n_layers} ({t_depth:.0f}% depth)")
        print(f"  → Potential to skip {t_savings:.0f}% of layers")

    if b_95_layer is not None:
        print(f"✓ Baseline reaches AUC≥0.95 at layer {b_95_layer}/{n_layers}")
    else:
        print(f"✗ Baseline NEVER reaches AUC≥0.95 (peaks at {max(results['baseline']['auc']):.4f})")

    # Accuracy comparison at key layers
    print("\n" + "-"*80)
    print("Accuracy at Key Exit Points")
    print("-"*80)

    key_layers = [15, 16, 17, 20, 24]
    print(f"{'Layer':<10} {'Baseline Acc':<15} {'Token Acc':<15}")
    print("-"*40)
    for layer in key_layers:
        b_acc = results["baseline"]["accuracy"][layer]
        t_acc = results["token"]["accuracy"][layer]
        print(f"{layer:<10} {b_acc*100:.1f}%{'':<9} {t_acc*100:.1f}%")

    # Save results
    output = {
        "metadata": data["metadata"],
        "early_exit_results": results,
        "thresholds": {
            "auc_0.90": {
                "baseline": find_earliest_threshold(results["baseline"]["auc"], 0.90),
                "token": find_earliest_threshold(results["token"]["auc"], 0.90)
            },
            "auc_0.95": {
                "baseline": find_earliest_threshold(results["baseline"]["auc"], 0.95),
                "token": find_earliest_threshold(results["token"]["auc"], 0.95)
            },
            "auc_0.98": {
                "baseline": find_earliest_threshold(results["baseline"]["auc"], 0.98),
                "token": find_earliest_threshold(results["token"]["auc"], 0.98)
            }
        },
        "headline": {
            "token_95_layer": t_95_layer,
            "token_depth_pct": (t_95_layer / n_layers) * 100 if t_95_layer else None,
            "potential_savings_pct": compute_potential_savings(t_95_layer, n_layers) * 100 if t_95_layer else None,
            "baseline_ever_reaches_95": b_95_layer is not None
        }
    }

    with open("early_exit_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to early_exit_results.json")

    return output


if __name__ == "__main__":
    main()
