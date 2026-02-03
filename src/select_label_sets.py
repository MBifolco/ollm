#!/usr/bin/env python3
"""
Select flat vs peaky label token sets under a controlled zero-shot probe.

Examples:
  python src/select_label_sets.py --k 4 --n_candidates 200 --template answer --space_mode space
  python src/select_label_sets.py --k 2 --exhaustive --template decision --template_text "DECISION: {opts}"
"""
from __future__ import annotations

import argparse
import json
import math
import random
from itertools import combinations
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATES = {
    "respond": "Respond with one letter: {opts}",
    "answer": "Answer: {opts}",
    "letter": "Letter: {opts}",
    "output": "Output: {opts}",
    "choice": "Choice: {opts}",
    "token": "Token: {opts}",
}

POOL_LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
POOL_DIGITS = [str(i) for i in range(10)]
POOL_SYMBOLS = [
    "!", "@", "#", "$", "%", "^", "&", "*", "?", "+", "=", "-", "_",
    "/", "\\", "|", "~", ":", ";", ".", ",", "<", ">", "(", ")", "[", "]", "{", "}",
]
POOL_EMOJI = [
    "😀", "😃", "😄", "😁", "😅", "😂", "🙂", "😉", "😊", "😍", "😎", "🤔",
    "👍", "👎", "🙏", "💡", "🔥", "⭐", "✅", "❌", "⚠️", "⚙️", "🧪", "🧠", "🚀",
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(example: dict, field: str, prefix: str, opts: list[str], template: str) -> str:
    scenario = example[field]
    rendered_prefix = prefix.format(scenario=scenario)
    return f"""{rendered_prefix}
{template.format(opts=', '.join(opts[:-1]) + ', or ' + opts[-1])}"""


@torch.no_grad()
def logits_for_prompt(tok, model, prompt: str, label_tokens: list[str]) -> np.ndarray:
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
    return probs


def tokenize_ok(tok, token: str) -> bool:
    ids = tok.encode(token, add_special_tokens=False)
    return len(ids) == 1


def expand_pool(pool: str, pool_types: list[str], pool_custom: str) -> list[str]:
    if pool and pool not in {"letters", "digits", "alnum"}:
        return [p.strip() for p in pool.split(",") if p.strip()]

    tokens = []
    for p in pool_types:
        if p == "letters":
            tokens.extend(POOL_LETTERS)
        elif p == "digits":
            tokens.extend(POOL_DIGITS)
        elif p == "symbols":
            tokens.extend(POOL_SYMBOLS)
        elif p == "emoji":
            tokens.extend(POOL_EMOJI)
        elif p == "alnum":
            tokens.extend(POOL_LETTERS + POOL_DIGITS)

    if pool_custom:
        tokens.extend([p.strip() for p in pool_custom.split(",") if p.strip()])

    # de-dupe preserving order
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def iter_candidate_sets(tokens: list[str], k: int, n_candidates: int, exhaustive: bool, seed: int):
    if exhaustive:
        for combo in combinations(tokens, k):
            yield list(combo)
        return

    rng = random.Random(seed)
    seen = set()
    max_attempts = n_candidates * 10
    attempts = 0
    while len(seen) < n_candidates and attempts < max_attempts:
        attempts += 1
        combo = tuple(sorted(rng.sample(tokens, k)))
        if combo in seen:
            continue
        seen.add(combo)
        yield list(combo)


def score_set(tok, model, samples, field, prefix, tokens, template, space_mode: str, include_reverse: bool):
    def eval_tokens(tok_list):
        ents = []
        maxps = []
        argmaxes = []
        for ex in samples:
            prompt = build_prompt(ex, field, prefix, tok_list, template)
            probs = logits_for_prompt(tok, model, prompt, tok_list)
            ents.append(-(probs * np.log(probs + 1e-12)).sum())
            maxps.append(probs.max())
            argmaxes.append(tok_list[int(np.argmax(probs))])
        return float(np.mean(ents)), float(np.mean(maxps)), Counter(argmaxes)

    results = []
    space_options = [False, True] if space_mode == "both" else [space_mode == "space"]
    order_options = [False, True] if include_reverse else [False]
    for add_space in space_options:
        tok_list = [(" " + t) if add_space else t for t in tokens]
        for reverse in order_options:
            order_list = list(reversed(tok_list)) if reverse else tok_list
            ent, maxp, counts = eval_tokens(order_list)
            results.append((ent, maxp, counts, add_space, reverse))

    mean_ent = float(np.mean([r[0] for r in results]))
    mean_maxp = float(np.mean([r[1] for r in results]))
    min_ent = float(np.min([r[0] for r in results]))
    max_maxp = float(np.max([r[1] for r in results]))
    return mean_ent, mean_maxp, min_ent, max_maxp


def main():
    parser = argparse.ArgumentParser(description="Select flat vs peaky label token sets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=str, default="data_k4_support/pilot_v4.jsonl")
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--pool", type=str, default="letters",
                        help="letters | digits | alnum | comma-separated list (overrides --pool_types)")
    parser.add_argument("--pool_types", type=str, default="letters",
                        help="Comma-separated subset of: letters,digits,emoji,symbols,alnum")
    parser.add_argument("--pool_custom", type=str, default="",
                        help="Comma-separated custom tokens to add to pool_types")
    parser.add_argument("--n_candidates", type=int, default=200)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--template", type=str, default="answer",
                        choices=["respond", "answer", "letter", "output", "choice", "token", "custom", "all"])
    parser.add_argument("--templates", type=str, default="",
                        help="Comma-separated templates to evaluate (overrides --template)")
    parser.add_argument("--template_text", type=str, default="",
                        help="Used when --template custom. Must include {opts}.")
    parser.add_argument("--field", type=str, default="scenario",
                        help="JSON field containing the scenario text")
    parser.add_argument("--prompt_prefix", type=str,
                        default="Classify the type of support in this scenario:\n\n{scenario}\n\n",
                        help="Prefix before template. Use {scenario} placeholder.")
    parser.add_argument("--space_mode", type=str, default="both",
                        choices=["both", "space", "nospace"])
    parser.add_argument("--include_reverse", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=str, default="",
                        help="Write results to JSON file")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    rng = random.Random(args.seed)
    rows = rows[:]  # copy
    rng.shuffle(rows)
    samples = rows[: args.n_examples]

    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    if args.templates:
        templates = []
        for name in [t.strip() for t in args.templates.split(",") if t.strip()]:
            if name == "custom":
                if "{opts}" not in args.template_text:
                    raise ValueError("--template_text must include {opts}")
                templates.append(("custom", args.template_text))
            else:
                templates.append((name, PROMPT_TEMPLATES[name]))
    elif args.template == "all":
        templates = list(PROMPT_TEMPLATES.items())
    elif args.template == "custom":
        if "{opts}" not in args.template_text:
            raise ValueError("--template_text must include {opts}")
        templates = [("custom", args.template_text)]
    else:
        templates = [(args.template, PROMPT_TEMPLATES[args.template])]

    pool_types = [p.strip() for p in args.pool_types.split(",") if p.strip()]
    pool = expand_pool(args.pool, pool_types, args.pool_custom)
    if args.space_mode == "space":
        valid = [t for t in pool if tokenize_ok(tok, " " + t)]
    elif args.space_mode == "nospace":
        valid = [t for t in pool if tokenize_ok(tok, t)]
    else:
        valid = [t for t in pool if tokenize_ok(tok, t) and tokenize_ok(tok, " " + t)]

    if len(valid) < args.k:
        raise ValueError(f"Pool too small after tokenization filter: {len(valid)} valid tokens")

    total_combos = math.comb(len(valid), args.k)
    exhaustive = args.exhaustive or total_combos <= args.n_candidates
    print(f"Valid tokens: {len(valid)} | k={args.k} | combos={total_combos} | exhaustive={exhaustive}")

    scored = []
    for tokens in iter_candidate_sets(valid, args.k, args.n_candidates, exhaustive, args.seed):
        ent_list = []
        maxp_list = []
        min_ent_list = []
        max_maxp_list = []
        for _, template in templates:
            mean_ent, mean_maxp, min_ent, max_maxp = score_set(
                tok, model, samples, args.field, args.prompt_prefix, tokens, template, args.space_mode, args.include_reverse
            )
            ent_list.append(mean_ent)
            maxp_list.append(mean_maxp)
            min_ent_list.append(min_ent)
            max_maxp_list.append(max_maxp)
        mean_ent = float(np.mean(ent_list))
        mean_maxp = float(np.mean(maxp_list))
        worst_ent = float(np.min(min_ent_list))
        worst_maxp = float(np.max(max_maxp_list))
        scored.append((mean_ent, mean_maxp, worst_ent, worst_maxp, tokens))

    flattest = sorted(scored, key=lambda x: (-x[2], x[3], -x[0], x[1]))[:5]
    peakiest = sorted(scored, key=lambda x: (x[2], -x[3], x[0], -x[1]))[:5]

    uniform_entropy = float(np.log(args.k))
    uniform_maxp = 1.0 / args.k
    print(f"\nUniform baseline: entropy {uniform_entropy:.4f} | maxp {uniform_maxp:.4f}\n")

    print("Flattest sets (best worst-case):")
    for mean_ent, mean_maxp, worst_ent, worst_maxp, tokens in flattest:
        print(
            f"  {tokens} | worst entropy {worst_ent:.4f} | worst maxp {worst_maxp:.4f} "
            f"| mean entropy {mean_ent:.4f} | mean maxp {mean_maxp:.4f}"
        )

    print("\nPeakiest sets (worst worst-case):")
    for mean_ent, mean_maxp, worst_ent, worst_maxp, tokens in peakiest:
        print(
            f"  {tokens} | worst entropy {worst_ent:.4f} | worst maxp {worst_maxp:.4f} "
            f"| mean entropy {mean_ent:.4f} | mean maxp {mean_maxp:.4f}"
        )

    if args.out_json:
        full_results = [
            {
                "tokens": tokens,
                "mean_entropy": mean_ent,
                "mean_maxp": mean_maxp,
                "worst_entropy": worst_ent,
                "worst_maxp": worst_maxp,
            }
            for mean_ent, mean_maxp, worst_ent, worst_maxp, tokens in scored
        ]
        top_choice = {
            "flattest": [
                {
                    "tokens": tokens,
                    "mean_entropy": mean_ent,
                    "mean_maxp": mean_maxp,
                    "worst_entropy": worst_ent,
                    "worst_maxp": worst_maxp,
                }
                for mean_ent, mean_maxp, worst_ent, worst_maxp, tokens in flattest[:2]
            ],
            "peakiest": [
                {
                    "tokens": tokens,
                    "mean_entropy": mean_ent,
                    "mean_maxp": mean_maxp,
                    "worst_entropy": worst_ent,
                    "worst_maxp": worst_maxp,
                }
                for mean_ent, mean_maxp, worst_ent, worst_maxp, tokens in peakiest[:2]
            ],
        }
        payload = {
            "config": {
                "model": args.model,
                "data": args.data,
                "n_examples": args.n_examples,
                "k": args.k,
                "pool": args.pool,
                "n_candidates": args.n_candidates,
                "exhaustive": exhaustive,
                "template": args.template,
                "templates": args.templates,
                "template_text": args.template_text,
                "field": args.field,
                "prompt_prefix": args.prompt_prefix,
                "space_mode": args.space_mode,
                "include_reverse": args.include_reverse,
                "seed": args.seed,
                "uniform_entropy": uniform_entropy,
                "uniform_maxp": uniform_maxp,
            },
            "full_results": full_results,
            "selected_sets": top_choice,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote results to {args.out_json}")


if __name__ == "__main__":
    main()
