"""
KSA Passive Auditing (Full Process).

Runs the full KSA pipeline for each trial:
  1. Each subset generates Yes/No → vote histogram
  2. Joint Exponential Mechanism selects output (K=1 → standard EM)

Uses Clopper-Pearson CI. Uses pure-DP estimator (exponential mechanism).

Usage:
    python ksa_passive.py --data_path /path/to/samsum/train.csv --dataset_name samsum
"""

import numpy as np
import math
import argparse
import os
import random
import time
import torch
from tqdm import tqdm

from auditing_utils import (
    load_hf_model, load_texts, build_membership_prompt, generate_response,
    parse_yes_no, clopper_pearson, pure_dp_eps_lower_bound,
)
import auditing_utils


def collect_vote_histogram(context_texts, query_text):
    """Query LLM for each context, return [yes_count, no_count]."""
    yes, no = 0, 0
    for text in context_texts:
        prompt = build_membership_prompt(text, query_text)
        response = generate_response(prompt)
        if parse_yes_no(response) == "Yes":
            yes += 1
        else:
            no += 1
    return np.array([yes, no], dtype=float)


def joint_em_select(histogram, epsilon):
    """Standard exponential mechanism (K=1). Returns 0=Yes, 1=No."""
    sensitivity = 1.0
    scores = histogram * epsilon / (2.0 * sensitivity)
    scores -= np.max(scores)
    probs = np.exp(scores)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def run_passive_audit(all_texts, canary_text, canary_index, num_exemplars,
                      n_trials, epsilons, delta, rng, f):
    remaining = [i for i in range(len(all_texts)) if i != canary_index]

    for epsilon in epsilons:
        print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

        # Run n_trials full processes
        bb_tp, bb_fp = 0, 0
        scores_with, scores_without = [], []

        for _ in tqdm(range(n_trials), desc=f"Passive (with, eps={epsilon})"):
            indices = rng.sample(remaining, num_exemplars - 1)
            contexts = [canary_text] + [all_texts[i] for i in indices]
            hist = collect_vote_histogram(contexts, canary_text)
            # Full process: EM selection
            sel = joint_em_select(hist, epsilon)
            if sel == 0:  # "Yes"
                bb_tp += 1
            # White-box: Gumbel-noised advantage
            noisy = hist * epsilon / 2.0 + np.random.gumbel(0, 1.0, 2)
            scores_with.append(noisy[0] - noisy[1])

        for _ in tqdm(range(n_trials), desc=f"Passive (without, eps={epsilon})"):
            indices = rng.sample(remaining, num_exemplars)
            contexts = [all_texts[i] for i in indices]
            hist = collect_vote_histogram(contexts, canary_text)
            sel = joint_em_select(hist, epsilon)
            if sel == 0:
                bb_fp += 1
            noisy = hist * epsilon / 2.0 + np.random.gumbel(0, 1.0, 2)
            scores_without.append(noisy[0] - noisy[1])

        confidence = 0.95

        # Black-box
        bb_fn = n_trials - bb_tp
        _, au_bb = clopper_pearson(bb_fp, n_trials, confidence)
        _, bu_bb = clopper_pearson(bb_fn, n_trials, confidence)
        eps_bb = pure_dp_eps_lower_bound(au_bb, bu_bb, delta)
        print(f"[Black-box] TPR={bb_tp/n_trials:.4f}, FPR={bb_fp/n_trials:.4f}", file=f)
        print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

        # White-box (threshold sweep)
        obs_with = np.sort(np.array(scores_with))
        obs_without = np.sort(np.array(scores_without))
        candidates = np.unique(np.concatenate((obs_with, obs_without)))

        max_eps = -float("inf")
        best_threshold = None

        for threshold in candidates:
            fp = n_trials - int(np.searchsorted(obs_without, threshold, side="right"))
            fn = int(np.searchsorted(obs_with, threshold, side="right"))
            _, au = clopper_pearson(fp, n_trials, confidence)
            _, bu = clopper_pearson(fn, n_trials, confidence)
            if au >= 1 or bu >= 1 or au <= 0 or bu <= 0:
                continue
            cur = pure_dp_eps_lower_bound(au, bu, delta)
            if cur > max_eps:
                max_eps, best_threshold = cur, threshold

        eps_wb = max(max_eps, 0.0)
        print(f"[White-box] eps_lower={eps_wb:.4f}", file=f)
        print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
        print(f"eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")


def main():
    parser = argparse.ArgumentParser(description="KSA Passive Auditing (Full Process)")
    parser.add_argument("--model_path", type=str, default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF")
    parser.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/Samsum/train.csv")
    parser.add_argument("--dataset_name", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_exemplars", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=2000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="results_ksa_passive")
    parser.add_argument("--output_prefix", type=str, default="ksa-passive")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, default="semantic_similarity")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.max_new_tokens_global = args.max_new_tokens
    load_hf_model(args.model_path)
    all_texts = load_texts(args.data_path, args.dataset_name)
    if not all_texts:
        print("Error: no texts loaded.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for run_id in range(args.num_runs):
        run_seed = None if args.seed == 0 else (args.seed + run_id)
        if run_seed is not None:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
        rng = random.Random(run_seed) if run_seed is not None else random

        if args.canary_mode == "fixed":
            idx = max(0, min(args.canary_index, len(all_texts) - 1))
        else:
            idx = rng.randrange(len(all_texts))
        canary = all_texts[idx]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("KSA Passive Auditing (Full Process)", file=f)
            print(f"Estimator: Pure DP (exponential mechanism)", file=f)
            print(f"Confidence: Clopper-Pearson 95%", file=f)
            print(f"n_trials={args.n_trials}, J={args.num_exemplars}", file=f)
            print("=" * 60, file=f)

            run_passive_audit(all_texts, canary, idx, args.num_exemplars,
                              args.n_trials, args.epsilons, args.delta, rng, f)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
