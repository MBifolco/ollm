#!/usr/bin/env python3
"""
Zero-shot categorical label probe with controls:
- 4-way sets only (cardinality matched)
- space vs no-space token variants
- prompt suffix variants (Respond/Answer/Letter)
- option order (normal vs reversed)
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SETS_4WAY = {
    "EPIS": ["E", "P", "I", "S"],
    "QJXZ": ["Q", "J", "X", "Z"],
    "RNAB": ["R", "N", "A", "B"],
}
SETS_2WAY = {
    "RN": ["R", "N"],
    "EP": ["E", "P"],
    "QJ": ["Q", "J"],
}

PROMPT_TEMPLATES = {
    "respond": "Respond with one letter: {opts}",
    "answer": "Answer: {opts}",
    "letter": "Letter: {opts}",
    "output": "Output: {opts}",
    "choice": "Choice: {opts}",
    "token": "Token: {opts}",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(example: dict, opts: list[str], template: str) -> str:
    return f"""Classify the type of support in this scenario:

{example['scenario']}

{template.format(opts=', '.join(opts[:-1]) + ', or ' + opts[-1])}"""


@torch.no_grad()
def logits_for_prompt(tok, model, prompt: str, label_tokens: list[str], topk: int | None = None):
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
    topk_list = None
    if topk and topk > 0:
        top_ids = torch.topk(logits, k=topk).indices.tolist()
        topk_list = [(tok.decode([i]), float(logits[i].item())) for i in top_ids]
    return probs, topk_list


def run_probe(tok, model, samples, set_name, tokens, template_name, template, add_space, reverse_order, topk: int | None):
    toks = [(" " + t) if add_space else t for t in tokens]
    if reverse_order:
        toks = list(reversed(toks))
    ents = []
    maxps = []
    argmaxes = []
    topk_examples = []
    for ex in samples:
        prompt = build_prompt(ex, toks, template)
        probs, topk_list = logits_for_prompt(tok, model, prompt, toks, topk=topk)
        ent = -(probs * np.log(probs + 1e-12)).sum()
        ents.append(ent)
        maxps.append(probs.max())
        argmaxes.append(toks[int(np.argmax(probs))])
        if topk_list is not None and len(topk_examples) == 0:
            topk_examples = topk_list
    return {
        "set": set_name,
        "template": template_name,
        "space": add_space,
        "reversed": reverse_order,
        "mean_entropy": float(np.mean(ents)),
        "mean_maxprob": float(np.mean(maxps)),
        "argmax_counts": dict(Counter(argmaxes)),
        "topk_example": topk_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot label probe with controls")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=str, default="data/k4_support/pilot_v4.jsonl")
    parser.add_argument("--n_examples", type=int, default=2)
    parser.add_argument("--template", type=str, default="all",
                        choices=["all", "respond", "answer", "letter", "output", "choice", "token"])
    parser.add_argument("--space_mode", type=str, default="both",
                        choices=["both", "space", "nospace"])
    parser.add_argument("--set_mode", type=str, default="4way",
                        choices=["4way", "2way", "all"],
                        help="Which label sets to evaluate")
    parser.add_argument("--include_reverse", action="store_true",
                        help="Include reversed option order")
    parser.add_argument("--topk", type=int, default=0,
                        help="If >0, print top-k unrestricted next-token logits for the first example")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    samples = rows[: args.n_examples]

    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    results = []
    if args.template == "all":
        templates = PROMPT_TEMPLATES.items()
    else:
        templates = [(args.template, PROMPT_TEMPLATES[args.template])]

    if args.space_mode == "both":
        space_options = [False, True]
    elif args.space_mode == "space":
        space_options = [True]
    else:
        space_options = [False]

    order_options = [False, True] if args.include_reverse else [False]

    if args.set_mode == "4way":
        sets = SETS_4WAY
    elif args.set_mode == "2way":
        sets = SETS_2WAY
    else:
        sets = {**SETS_4WAY, **SETS_2WAY}

    for set_name, tokens in sets.items():
        for template_name, template in templates:
            for add_space in space_options:
                for reverse_order in order_options:
                    results.append(
                        run_probe(tok, model, samples, set_name, tokens, template_name, template, add_space, reverse_order, args.topk)
                    )

    # Print summary grouped by set/template/space/reversed
    for r in results:
        space_tag = "space" if r["space"] else "nospace"
        order_tag = "reversed" if r["reversed"] else "normal"
        k = len(SETS_4WAY.get(r["set"], SETS_2WAY.get(r["set"])))
        uniform_entropy = float(np.log(k))
        uniform_maxp = 1.0 / k
        print(
            f"{r['set']} | {r['template']} | {space_tag} | {order_tag} "
            f"=> entropy {r['mean_entropy']:.4f} (uniform {uniform_entropy:.4f}) "
            f"| maxp {r['mean_maxprob']:.4f} (uniform {uniform_maxp:.4f}) "
            f"| argmax {r['argmax_counts']}"
        )
        if r["topk_example"]:
            topk_str = ", ".join([f"{t}:{v:.2f}" for t, v in r["topk_example"]])
            print(f"  topk {r['set']} {r['template']} {space_tag} {order_tag} => {topk_str}")


if __name__ == "__main__":
    main()
