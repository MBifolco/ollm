#!/usr/bin/env python3
"""
Zero-shot categorical label probe + embedding geometry.

Runs two quick diagnostics on a base model (no fine-tuning):
1) Embedding geometry of candidate label sets (cosine similarity stats).
2) Zero-shot categorical prompt probe (entropy + max prob over label tokens).

Usage:
    python src/zero_shot_label_probe.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/k4_support/pilot_v4.jsonl \
        --n_examples 2
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_SETS = {
    "EPIS": ["E", "P", "I", "S"],
    "RN": ["R", "N"],
    "QJXZ": ["Q", "J", "X", "Z"],
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def set_geometry(tok, emb, label: str, tokens: list[str]) -> None:
    ids = [tok.encode(t, add_special_tokens=False) for t in tokens]
    if any(len(i) != 1 for i in ids):
        print(f"{label}: non-single tokens present: {ids}")
        return
    ids = [i[0] for i in ids]
    vecs = [emb[i] for i in ids]
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine(vecs[i], vecs[j]))
    sims = np.array(sims)
    print(f"{label} tokens {tokens} ids {ids} | mean cos {sims.mean():.4f} std {sims.std():.4f} min {sims.min():.4f} max {sims.max():.4f}")


def prompt_for_example(example: dict, label_tokens: list[str]) -> str:
    return f"""Classify the type of support in this scenario:

{example['scenario']}

Respond with one letter: {', '.join(label_tokens[:-1])}, or {label_tokens[-1]}"""


@torch.no_grad()
def logits_for_prompt(tok, model, prompt: str, label_tokens: list[str]) -> tuple[np.ndarray, float]:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model(input_ids)
    logits = out.logits[0, -1, :].float()

    cand_ids = [tok.encode(t, add_special_tokens=False) for t in label_tokens]
    if any(len(i) != 1 for i in cand_ids):
        raise ValueError(f"Non-single tokens in {label_tokens}: {cand_ids}")
    cand_ids = [i[0] for i in cand_ids]
    cand_logits = logits[cand_ids].cpu().numpy()
    exp = np.exp(cand_logits - cand_logits.max())
    probs = exp / exp.sum()
    ent = -(probs * np.log(probs + 1e-12)).sum()
    return probs, ent


def main():
    parser = argparse.ArgumentParser(description="Zero-shot categorical label probe")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=str, default="data/k4_support/pilot_v4.jsonl")
    parser.add_argument("--n_examples", type=int, default=2)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    samples = rows[: args.n_examples]

    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    # Embedding geometry
    emb = model.model.embed_tokens.weight.detach().float()
    print("\nEmbedding geometry:")
    for name, tokens in DEFAULT_SETS.items():
        set_geometry(tok, emb, name, tokens)

    # Zero-shot categorical prompt probe
    print(f"\nZero-shot categorical prompt probe (avg over {len(samples)} examples):")
    for name, tokens in DEFAULT_SETS.items():
        ents = []
        maxps = []
        max_tokens = []
        for ex in samples:
            p, ent = logits_for_prompt(tok, model, prompt_for_example(ex, tokens), tokens)
            ents.append(ent)
            maxps.append(p.max())
            max_tokens.append(tokens[int(np.argmax(p))])
        print(f"{name}: mean entropy {np.mean(ents):.4f} | mean max prob {np.mean(maxps):.4f}")
        if name == "RN":
            print(f"  RN argmax tokens (per example): {max_tokens}")


if __name__ == "__main__":
    main()
