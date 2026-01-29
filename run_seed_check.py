#!/usr/bin/env python3
"""Run Track 1 training and evaluation across multiple seeds."""

import subprocess
import json
import os

SEEDS = [0, 1, 2]

def run_training(seed, model_type):
    """Train a model with a specific seed."""
    output_dir = f"models/{model_type}_track1_seed{seed}"
    
    if model_type == "baseline":
        script = "src/train_baseline.py"
    else:
        script = "src/train_internal_token.py"
    
    # Add seed to training args by modifying the script call
    cmd = f"HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 {script} --output_dir {output_dir}"
    
    print(f"\n{'='*50}")
    print(f"Training {model_type} with seed {seed}")
    print(f"{'='*50}")
    
    # We need to modify the training to use the seed
    # For now, we'll create a wrapper that sets the seed
    wrapper = f'''
import sys
sys.argv = ["{script}", "--output_dir", "{output_dir}"]

import random
import numpy as np
import torch
random.seed({seed})
np.random.seed({seed})
torch.manual_seed({seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({seed})

# Now import and run the training
exec(open("{script}").read())
'''
    
    with open(f"_train_seed{seed}_{model_type}.py", "w") as f:
        f.write(wrapper)
    
    result = subprocess.run(
        f"HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 _train_seed{seed}_{model_type}.py",
        shell=True, capture_output=True, text=True
    )
    
    os.remove(f"_train_seed{seed}_{model_type}.py")
    
    if result.returncode != 0:
        print(f"Training failed: {result.stderr[-500:]}")
        return False
    return True

def run_probing(seed):
    """Run Track 1 probing for a specific seed."""
    from src.probe_track1 import evaluate_track1
    
    baseline_path = f"models/baseline_track1_seed{seed}"
    internal_path = f"models/internal_token_track1_seed{seed}"
    
    results = {}
    
    # Baseline
    _, baseline_metrics = evaluate_track1(
        model_path=baseline_path,
        test_data_path="data/test.jsonl",
        model_type="baseline"
    )
    results["baseline"] = baseline_metrics
    
    # Internal token
    internal_results, internal_metrics = evaluate_track1(
        model_path=internal_path,
        test_data_path="data/test.jsonl",
        model_type="internal_token"
    )
    results["internal_token"] = internal_metrics
    
    # Compute additional metrics
    rom_correct = sum(1 for r in internal_results if r.ground_truth == "romantic" and r.correct)
    rom_total = sum(1 for r in internal_results if r.ground_truth == "romantic")
    nonrom_correct = sum(1 for r in internal_results if r.ground_truth == "non-romantic" and r.correct)
    nonrom_total = sum(1 for r in internal_results if r.ground_truth == "non-romantic")
    
    results["internal_token"]["rom_recall"] = rom_correct / rom_total if rom_total > 0 else 0
    results["internal_token"]["nonrom_recall"] = nonrom_correct / nonrom_total if nonrom_total > 0 else 0
    
    # AUC
    rom_probs = [(r.p_rom_token, r.ground_truth == "romantic") for r in internal_results]
    rom_probs.sort(reverse=True)
    n_pos = sum(1 for _, label in rom_probs if label)
    n_neg = len(rom_probs) - n_pos
    concordant = 0
    for i, (prob_i, label_i) in enumerate(rom_probs):
        if label_i:
            concordant += sum(1 for j in range(i+1, len(rom_probs)) if not rom_probs[j][1])
    results["internal_token"]["auc"] = concordant / (n_pos * n_neg) if n_pos * n_neg > 0 else 0
    
    return results

if __name__ == "__main__":
    all_results = {}
    
    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"SEED {seed}")
        print(f"{'#'*60}")
        
        # Train both models
        run_training(seed, "baseline")
        run_training(seed, "internal_token")
        
        # Run probing
        results = run_probing(seed)
        all_results[f"seed_{seed}"] = results
    
    # Save results
    with open("seed_check_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SEED CHECK SUMMARY")
    print("="*70)
    
    print(f"\n{'Seed':<6} {'Baseline':>10} {'Token Acc':>10} {'ROM Rec':>10} {'NONROM Rec':>10} {'AUC':>10}")
    print("-"*70)
    
    for seed in SEEDS:
        r = all_results[f"seed_{seed}"]
        b_acc = r["baseline"]["accuracy"]
        t_acc = r["internal_token"]["accuracy"]
        rom_rec = r["internal_token"]["rom_recall"]
        nonrom_rec = r["internal_token"]["nonrom_recall"]
        auc = r["internal_token"]["auc"]
        print(f"{seed:<6} {b_acc:>10.2%} {t_acc:>10.2%} {rom_rec:>10.2%} {nonrom_rec:>10.2%} {auc:>10.4f}")
    
    # Compute mean and std
    import numpy as np
    baseline_accs = [all_results[f"seed_{s}"]["baseline"]["accuracy"] for s in SEEDS]
    token_accs = [all_results[f"seed_{s}"]["internal_token"]["accuracy"] for s in SEEDS]
    rom_recs = [all_results[f"seed_{s}"]["internal_token"]["rom_recall"] for s in SEEDS]
    nonrom_recs = [all_results[f"seed_{s}"]["internal_token"]["nonrom_recall"] for s in SEEDS]
    aucs = [all_results[f"seed_{s}"]["internal_token"]["auc"] for s in SEEDS]
    
    print("-"*70)
    print(f"{'Mean':<6} {np.mean(baseline_accs):>10.2%} {np.mean(token_accs):>10.2%} {np.mean(rom_recs):>10.2%} {np.mean(nonrom_recs):>10.2%} {np.mean(aucs):>10.4f}")
    print(f"{'Std':<6} {np.std(baseline_accs):>10.2%} {np.std(token_accs):>10.2%} {np.std(rom_recs):>10.2%} {np.std(nonrom_recs):>10.2%} {np.std(aucs):>10.4f}")
