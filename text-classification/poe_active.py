"""
PoE Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean utility vectors via real LLM log-prob calls.
Stage 2: Bootstrap n_sample noisy utilities by resampling + Gumbel noise.
Stage 3: Compute TPR/FPR with Bernstein (snapshot) + Clopper-Pearson (MC),
         then convert to epsilon via pure-DP estimator.

Usage:
    python poe_active.py --data_path /path/to/data.csv --epsilons 1 2 4 8 16
"""

import numpy as np
import math
import argparse
import os
import random
import time
import json
import torch
from tqdm import tqdm

from auditing_utils import (
    load_hf_model, load_texts_from_csv, build_poe_expert_prompt,
    get_label_token_ids, get_log_probs_for_labels,
    clopper_pearson, pure_dp_eps_lower_bound,
    DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PoE Mechanism
# ==============================================================================
def clip_log_probs(log_probs: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(log_probs, -gamma, 0.0)


class PoEClassifier:
    def __init__(self, epsilon, delta, gamma=2.0):
        self.class_labels = ["Yes", "No"]
        self.num_classes = 2
        self.label_to_id = {"Yes": 0, "No": 1}
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.label_token_ids = None

    def init_label_tokens(self):
        self.label_token_ids = get_label_token_ids(self.class_labels)

    def get_clean_utility_vector(self, context_texts, query_text):
        utility = np.zeros(self.num_classes)
        for text in context_texts:
            prompt = build_poe_expert_prompt(text, query_text)
            log_probs = get_log_probs_for_labels(prompt, self.label_token_ids)
            utility += clip_log_probs(log_probs, self.gamma)
        return utility


# ==============================================================================
# Stage 1: Collect clean utility vectors
# ==============================================================================
def collect_snapshots(poe, canary_text, pool_texts, canary_index,
                      num_exemplars, n_llm, rng, f,
                      fixed_base_indices=None, fixed_replace_index=None):
    """
    Collect clean utility vectors.

    If fixed_base_indices and fixed_replace_index are provided, use fixed context
    (deterministic -- only one LLM call needed, then replicated n_llm times).
    Otherwise, randomly sample context each time.
    """
    if fixed_base_indices is not None and fixed_replace_index is not None:
        # Fixed context mode: compute once, replicate
        # Only use num_exemplars-1 base indices (so total experts = num_exemplars)
        used_base_indices = fixed_base_indices[:num_exemplars - 1]
        base_texts = [pool_texts[i] for i in used_base_indices]
        replace_text = pool_texts[fixed_replace_index]

        contexts_with = [canary_text] + base_texts
        contexts_without = [replace_text] + base_texts

        print(f"[Fixed context] base_indices={used_base_indices} (J={num_exemplars})", file=f)
        print(f"[Fixed context] replace_index={fixed_replace_index}", file=f)

        vec_with = poe.get_clean_utility_vector(contexts_with, canary_text)
        vec_without = poe.get_clean_utility_vector(contexts_without, canary_text)

        print(f"Utility (with):    {vec_with}", file=f)
        print(f"Utility (without): {vec_without}", file=f)

        vectors_with = [vec_with] * n_llm
        vectors_without = [vec_without] * n_llm
    else:
        # Random context mode (original behavior)
        remaining = [i for i in range(len(pool_texts)) if i != canary_index]

        vectors_without = []
        for _ in tqdm(range(n_llm), desc="Snapshots (without canary)"):
            indices = rng.sample(remaining, num_exemplars)
            contexts = [pool_texts[i] for i in indices]
            vectors_without.append(poe.get_clean_utility_vector(contexts, canary_text))

        vectors_with = []
        for _ in tqdm(range(n_llm), desc="Snapshots (with canary)"):
            indices = rng.sample(remaining, num_exemplars - 1)
            contexts = [canary_text] + [pool_texts[i] for i in indices]
            vectors_with.append(poe.get_clean_utility_vector(contexts, canary_text))

    print(f"Mean utility (with): {np.mean(vectors_with, axis=0)}", file=f)
    print(f"Mean utility (without): {np.mean(vectors_without, axis=0)}", file=f)
    return vectors_with, vectors_without


# ==============================================================================
# Stage 2: Bootstrap with Gumbel noise
# ==============================================================================
def bootstrap_simulate_gumbel(collected_vectors, epsilon, gamma, num_classes, n_sample):
    gumbel_scale = 2.0 * gamma / epsilon
    noisy = []
    for _ in range(n_sample):
        clean = random.choice(collected_vectors)
        noise = np.random.gumbel(0.0, gumbel_scale, num_classes)
        noisy.append(clean + noise)
    return noisy


# ==============================================================================
# Stage 3: Audit with two-layer error bound
# ==============================================================================
def run_active_audit(noisy_with, noisy_without, n_llm, n_sample,
                     delta, gamma_conf, f, snapshot_gamma_frac=0.5):
    canary_class_id = 0
    T_audit = len(noisy_with)

    # Advantage scores
    def compute_scores(utilities):
        scores = np.zeros(len(utilities))
        for i, util in enumerate(utilities):
            mx = max(util[j] for j in range(len(util)) if j != canary_class_id)
            scores[i] = util[canary_class_id] - mx
        return scores

    scores_with = compute_scores(noisy_with)
    scores_without = compute_scores(noisy_without)

    # --- Two-layer error: Bernstein (snapshot) + Clopper-Pearson (MC) ---
    # For PoE, per-snapshot variance is ~0 (Table 2: Var(h(X))=0 for PoE)
    sigma2_with = 0.0
    sigma2_without = 0.0

    # Snapshot-phase Bernstein term (no MC Hoeffding -- replaced by Clopper-Pearson)
    gamma_snap = gamma_conf * snapshot_gamma_frac
    gamma_mc   = gamma_conf * (1.0 - snapshot_gamma_frac)
    log_snap = math.log(2.0 / gamma_snap)

    bern_with = (math.sqrt(2.0 * sigma2_with * log_snap / n_llm)
                 + 7.0 * log_snap / (3.0 * (n_llm - 1)))
    bern_without = (math.sqrt(2.0 * sigma2_without * log_snap / n_llm)
                    + 7.0 * log_snap / (3.0 * (n_llm - 1)))

    # Clopper-Pearson confidence level for the MC phase
    cp_confidence = 1.0 - gamma_mc

    print(f"\n--- Error Bounds (gamma={gamma_conf}, snapshot_gamma_frac={snapshot_gamma_frac}) ---", file=f)
    print(f"  Bernstein_1 (snapshot): {bern_with:.6f}", file=f)
    print(f"  Bernstein_0 (snapshot): {bern_without:.6f}", file=f)
    print(f"  Clopper-Pearson confidence: {cp_confidence}", file=f)

    # Helper: compute TPR lower / FPR upper using CP + Bernstein
    def get_tpr_low_fpr_up(tp_count, fp_count, n):
        # Clopper-Pearson interval on the MC counts
        cp_tpr_lo, _ = clopper_pearson(tp_count, n, cp_confidence)
        _, cp_fpr_hi = clopper_pearson(fp_count, n, cp_confidence)
        # Add Bernstein snapshot correction
        tpr_low = max(0.0, cp_tpr_lo - bern_with)
        fpr_up  = min(1.0, cp_fpr_hi + bern_without)
        return tpr_low, fpr_up

    # === Black-box (threshold at 0) ===
    tp_count = int(np.sum(scores_with > 0))
    fp_count = int(np.sum(scores_without > 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low, fpr_up = get_tpr_low_fpr_up(tp_count, fp_count, T_audit)
    fnr_up = 1.0 - tpr_low

    eps_bb = pure_dp_eps_lower_bound(fpr_up, fnr_up, delta)
    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f}, FPR_hat={fpr_hat:.6f}", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.6f}", file=f)
    print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds) ===
    obs_with = np.sort(scores_with)
    obs_without = np.sort(scores_without)
    candidates = np.unique(np.concatenate((obs_with, obs_without)))

    max_eps = -float("inf")
    best_threshold = None

    for threshold in candidates:
        fp = T_audit - int(np.searchsorted(obs_without, threshold, side="right"))
        fn = int(np.searchsorted(obs_with, threshold, side="right"))
        tp = T_audit - fn

        tpr_low_t, fpr_up_t = get_tpr_low_fpr_up(tp, fp, T_audit)
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
    parser = argparse.ArgumentParser(description="PoE Active Auditing (Bootstrap + Bernstein+CP Error)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_exemplars", type=int, default=1, help="J: number of experts")
    parser.add_argument("--n_llm", type=int, default=2000, help="Number of clean snapshots")
    parser.add_argument("--n_sample", type=int, default=400000, help="Bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_clip", type=float, default=4.5, help="Log-prob clipping bound")
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--snapshot_gamma_frac", type=float, default=0.5,
                        help="Fraction of gamma allocated to snapshot phase (>0.5 = tighter snapshot bound)")
    parser.add_argument("--output_dir", type=str, default="results_poe_active")
    parser.add_argument("--output_prefix", type=str, default="poe-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack_mode", type=str, choices=["direct", "semantic_similarity"], default="direct")
    parser.add_argument("--context_file", type=str, default=None,
                        help="JSON file from poe_context_search.py with pre-selected contexts")
    parser.add_argument("--use_best_canary", action="store_true",
                        help="Use only the best canary (highest gap) from context_file for all runs")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Logit temperature (<1.0 sharpens distribution)")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.logit_temperature = args.temperature
    load_hf_model(args.model_path)
    all_texts = load_texts_from_csv(args.data_path, args.text_col, args.delimiter, not args.no_header)
    if not all_texts:
        print("Error: no texts loaded.")
        return

    # Load pre-searched contexts if provided
    context_entries = None
    if args.context_file:
        with open(args.context_file, "r") as cf:
            context_entries = json.load(cf)
        print(f"Loaded {len(context_entries)} pre-searched contexts from {args.context_file}")

        if args.use_best_canary:
            # Pick only the canary with the highest gap
            best_entry = max(context_entries, key=lambda e: e["best_gap"])
            print(f"[use_best_canary] Selected canary idx={best_entry['canary_index']} "
                  f"with gap={best_entry['best_gap']:.4f}")
            context_entries = [best_entry]

    os.makedirs(args.output_dir, exist_ok=True)

    num_runs = len(context_entries) * args.num_runs if context_entries else args.num_runs

    for run_id in range(num_runs):
        run_seed = None if args.seed == 0 else (args.seed + run_id)
        if run_seed is not None:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

        rng = random.Random(run_seed) if run_seed is not None else random

        # Determine canary and context
        fixed_base_indices = None
        fixed_replace_index = None

        if context_entries:
            entry = context_entries[run_id % len(context_entries)]
            idx = entry["canary_index"]
            fixed_base_indices = entry["base_indices"]
            fixed_replace_index = entry["replace_index"]
        elif args.canary_mode == "fixed":
            idx = max(0, min(args.canary_index, len(all_texts) - 1))
        else:
            idx = rng.randrange(len(all_texts))

        canary_text = all_texts[idx]
        context_mode = "fixed" if context_entries else "random"

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PoE Active Auditing (Bootstrap + Bernstein+CP Error Bound)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Mechanism: Exponential (Gumbel noise)", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Clopper-Pearson (MC)", file=f)
            print(f"Estimator: Pure DP", file=f)
            print(f"Context mode: {context_mode}", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, J={args.num_exemplars}", file=f)
            print("=" * 60, file=f)

            poe = PoEClassifier(1.0, args.delta, args.gamma_clip)
            poe.init_label_tokens()

            t1_start = time.time()
            vecs_with, vecs_without = collect_snapshots(
                poe, canary_text, all_texts, idx,
                args.num_exemplars, args.n_llm, rng, f,
                fixed_base_indices=fixed_base_indices,
                fixed_replace_index=fixed_replace_index,
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Stage 1: {t1_time:.1f}s", file=f)

            for epsilon in args.epsilons:
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

                t2_start = time.time()
                noisy_with = bootstrap_simulate_gumbel(
                    vecs_with, epsilon, args.gamma_clip, poe.num_classes, args.n_sample
                )
                noisy_without = bootstrap_simulate_gumbel(
                    vecs_without, epsilon, args.gamma_clip, poe.num_classes, args.n_sample
                )
                t2_time = time.time() - t2_start

                t3_start = time.time()
                eps_bb, eps_wb = run_active_audit(
                    noisy_with, noisy_without,
                    n_llm=args.n_llm, n_sample=args.n_sample,
                    delta=args.delta, gamma_conf=args.gamma_conf, f=f,
                    snapshot_gamma_frac=args.snapshot_gamma_frac
                )
                t3_time = time.time() - t3_start

                total = t1_time + t2_time + t3_time
                print(f"\n[TIMING] Stage 2: {t2_time:.1f}s, Stage 3: {t3_time:.1f}s", file=f)
                print(f"[TIMING] Total: {total:.1f}s, per-sample: {total/args.n_sample*1000:.4f}ms", file=f)
                print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
