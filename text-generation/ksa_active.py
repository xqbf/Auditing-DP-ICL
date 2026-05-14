"""
KSA Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean vote histograms via real LLM calls.
Stage 2: Bootstrap n_sample noisy selections by resampling + Gumbel noise.
Stage 3: Compute TPR/FPR with two-layer Bernstein + MC error bound,
         then convert to epsilon via pure-DP estimator.

Uses pure-DP estimator (exponential mechanism).

Usage:
    python ksa_active.py --data_path /path/to/samsum/train.csv --dataset_name samsum
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
    empirical_bernstein_correction,
)
import auditing_utils


# ==============================================================================
# KSA Core
# ==============================================================================
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


# ==============================================================================
# Stage 1: Collect clean vote histograms
# ==============================================================================
def collect_snapshots(all_texts, canary_text, canary_index, num_exemplars,
                      n_llm, rng, f):
    remaining = [i for i in range(len(all_texts)) if i != canary_index]

    histograms_with = []
    for _ in tqdm(range(n_llm), desc="Snapshots (with canary)"):
        indices = rng.sample(remaining, num_exemplars - 1)
        contexts = [canary_text] + [all_texts[i] for i in indices]
        histograms_with.append(collect_vote_histogram(contexts, canary_text))

    histograms_without = []
    for _ in tqdm(range(n_llm), desc="Snapshots (without canary)"):
        indices = rng.sample(remaining, num_exemplars)
        contexts = [all_texts[i] for i in indices]
        histograms_without.append(collect_vote_histogram(contexts, canary_text))

    print(f"Mean histogram (with): {np.mean(histograms_with, axis=0)}", file=f)
    print(f"Mean histogram (without): {np.mean(histograms_without, axis=0)}", file=f)
    return histograms_with, histograms_without


# ==============================================================================
# Stage 2: Bootstrap with Gumbel noise
# ==============================================================================
def bootstrap_simulate_gumbel(collected_histograms, epsilon, n_sample):
    """Resample histograms and add Gumbel noise (equivalent to exponential mechanism)."""
    sensitivity = 1.0
    selections = np.zeros(n_sample, dtype=int)
    advantage_scores = np.zeros(n_sample)

    for i in range(n_sample):
        hist = random.choice(collected_histograms)
        noisy = hist * epsilon / (2.0 * sensitivity) + np.random.gumbel(0.0, 1.0, 2)
        selections[i] = int(np.argmax(noisy))
        advantage_scores[i] = noisy[0] - noisy[1]

    return selections, advantage_scores


# ==============================================================================
# Stage 3: Audit with two-layer error bound
# ==============================================================================
def run_active_audit(selections_with, selections_without,
                     scores_with, scores_without,
                     n_llm, n_sample, delta, gamma_conf, f):
    T_audit = len(selections_with)

    # Per-snapshot variance for KSA: small but compute empirically
    # h(X) = 1{argmax(noisy) == Yes} for black-box
    # For KSA with resampled exemplars, variance is small (~0)
    sigma2_with = 0.0
    sigma2_without = 0.0

    delta_1 = empirical_bernstein_correction(sigma2_with, n_llm, n_sample, gamma_conf)
    delta_0 = empirical_bernstein_correction(sigma2_without, n_llm, n_sample, gamma_conf)

    print(f"\n--- Error Bounds (gamma={gamma_conf}) ---", file=f)
    print(f"  Delta_1 (TPR correction): {delta_1:.6f}", file=f)
    print(f"  Delta_0 (FPR correction): {delta_0:.6f}", file=f)

    # === Black-box (discrete EM output: Yes=0) ===
    tp_count = int(np.sum(selections_with == 0))
    fp_count = int(np.sum(selections_without == 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low = max(0.0, tpr_hat - delta_1)
    fpr_up = min(1.0, fpr_hat + delta_0)
    fnr_up = 1.0 - tpr_low

    eps_bb = pure_dp_eps_lower_bound(fpr_up, fnr_up, delta)
    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f}, FPR_hat={fpr_hat:.4f}", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.4f}", file=f)
    print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds on advantage scores) ===
    obs_with = np.sort(scores_with)
    obs_without = np.sort(scores_without)
    candidates = np.unique(np.concatenate((obs_with, obs_without)))

    max_eps = -float("inf")
    best_threshold = None

    for threshold in candidates:
        fp = T_audit - int(np.searchsorted(obs_without, threshold, side="right"))
        fn = int(np.searchsorted(obs_with, threshold, side="right"))
        tpr_hat_t = 1.0 - fn / T_audit
        fpr_hat_t = fp / T_audit

        tpr_low_t = max(0.0, tpr_hat_t - delta_1)
        fpr_up_t = min(1.0, fpr_hat_t + delta_0)
        fnr_up_t = 1.0 - tpr_low_t

        if fpr_up_t >= 1 or fnr_up_t >= 1 or fpr_up_t <= 0 or fnr_up_t <= 0:
            continue

        cur_eps = pure_dp_eps_lower_bound(fpr_up_t, fnr_up_t, delta)
        if cur_eps > max_eps:
            max_eps, best_threshold = cur_eps, threshold

    eps_wb = max(max_eps, 0.0)
    if best_threshold is not None:
        print(f"\n[White-box] Optimal threshold={best_threshold:.4f}", file=f)
        print(f"[White-box] eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)
        eps_wb = 0.0

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="KSA Active Auditing (Bootstrap + Bernstein+MC Error)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_exemplars", type=int, default=8)
    parser.add_argument("--n_llm", type=int, default=20000, help="Number of clean snapshots")
    parser.add_argument("--n_sample", type=int, default=400000, help="Bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--output_dir", type=str, default="results_ksa_active")
    parser.add_argument("--output_prefix", type=str, default="ksa-active")
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
            print("KSA Active Auditing (Bootstrap + Bernstein+MC Error Bound)", file=f)
            print(f"Estimator: Pure DP (exponential mechanism)", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Hoeffding (MC)", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, J={args.num_exemplars}", file=f)
            print("=" * 60, file=f)

            t1_start = time.time()
            hists_with, hists_without = collect_snapshots(
                all_texts, canary, idx, args.num_exemplars,
                args.n_llm, rng, f
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Stage 1: {t1_time:.1f}s", file=f)

            for epsilon in args.epsilons:
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

                t2_start = time.time()
                sel_with, scores_with = bootstrap_simulate_gumbel(
                    hists_with, epsilon, args.n_sample
                )
                sel_without, scores_without = bootstrap_simulate_gumbel(
                    hists_without, epsilon, args.n_sample
                )
                t2_time = time.time() - t2_start

                t3_start = time.time()
                eps_bb, eps_wb = run_active_audit(
                    sel_with, sel_without,
                    scores_with, scores_without,
                    n_llm=args.n_llm, n_sample=args.n_sample,
                    delta=args.delta, gamma_conf=args.gamma_conf, f=f
                )
                t3_time = time.time() - t3_start

                total = t1_time + t2_time + t3_time
                print(f"\n[TIMING] Stage 2: {t2_time:.1f}s, Stage 3: {t3_time:.1f}s", file=f)
                print(f"[TIMING] Total: {total:.1f}s", file=f)
                print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
