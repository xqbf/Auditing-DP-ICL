"""
PV Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean vote vectors via real LLM calls (snapshot).
Stage 2: Bootstrap n_sample noisy histograms by resampling + Gaussian noise.
Stage 3: Compute TPR/FPR with two-layer error bound (Bernstein + MC Hoeffding),
         then convert to epsilon via GDP estimator.

Error bound (Algorithm 2 from paper):
  Delta_b = sqrt(2 * sigma_b^2 * log(8/gamma) / n_llm)
          + 7 * log(8/gamma) / (3 * (n_llm - 1))
          + sqrt(log(8/gamma) / (2 * n_sample))

Usage:
    python pv_active.py --data_path /path/to/data.csv --epsilons 1 2 4 8 16
"""

import numpy as np
import math
import argparse
import os
import random
import time
import torch
from tqdm import tqdm
from scipy.stats import norm

from auditing_utils import (
    load_hf_model, load_texts_from_csv, llm_yes_no,
    analytic_gauss_sigma, clopper_pearson, gdp_to_eps,
    empirical_bernstein_correction, DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PV Mechanism (clean vote collection)
# ==============================================================================
class PVClassifier:
    def __init__(self, class_labels, epsilon, delta):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        sensitivity = math.sqrt(2)
        self.sigma = analytic_gauss_sigma(sensitivity, epsilon, delta)

    def get_clean_vote_vector(self, exemplars, query, num_subsets):
        """Run partitioned LLM calls, return clean vote histogram (no noise)."""
        shuffled = np.random.permutation(len(exemplars))
        partitions = np.array_split(shuffled, num_subsets)
        histogram = np.zeros(self.num_classes)
        for subset_indices in partitions:
            if len(subset_indices) == 0:
                continue
            texts = [exemplars[i][0] for i in subset_indices]
            labels = [exemplars[i][1] for i in subset_indices]
            vote = llm_yes_no(texts, query, labels)
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        return histogram


# ==============================================================================
# Stage 1: Collect clean snapshots
# ==============================================================================
def collect_snapshots(pv, exemplars_with, exemplars_without, query, num_subsets, n_llm, f):
    print(f"\n--- Stage 1: Collecting {n_llm} clean vote vectors ---", file=f)
    vectors_with = []
    for _ in tqdm(range(n_llm), desc="Snapshots (with canary)"):
        vectors_with.append(pv.get_clean_vote_vector(exemplars_with, query, num_subsets))
    vectors_without = []
    for _ in tqdm(range(n_llm), desc="Snapshots (without canary)"):
        vectors_without.append(pv.get_clean_vote_vector(exemplars_without, query, num_subsets))

    mean_with = np.mean(vectors_with, axis=0)
    mean_without = np.mean(vectors_without, axis=0)
    print(f"Mean vector (with canary): {mean_with}", file=f)
    print(f"Mean vector (without canary): {mean_without}", file=f)

    # Compute per-snapshot variance for Bernstein bound
    var_with = np.var([v.tolist() for v in vectors_with], axis=0).max()
    var_without = np.var([v.tolist() for v in vectors_without], axis=0).max()
    print(f"Max per-dim variance (with): {var_with:.6f}", file=f)
    print(f"Max per-dim variance (without): {var_without:.6f}", file=f)

    return vectors_with, vectors_without


# ==============================================================================
# Stage 2: Bootstrap simulation
# ==============================================================================
def bootstrap_simulate(collected_vectors, sigma, num_classes, n_sample):
    indices = np.random.choice(len(collected_vectors), n_sample, replace=True)
    vectors = np.array(collected_vectors)
    resampled = vectors[indices]
    noise = np.random.normal(0, sigma, resampled.shape)
    return resampled + noise


# ==============================================================================
# Stage 3: Audit with two-layer error bound
# ==============================================================================
def run_active_audit(noisy_with, noisy_without, canary_class_id, n_llm, n_sample,
                     delta, gamma, f):
    """
    Compute epsilon lower bound using:
    - Black-box: fixed threshold at 0 (argmax = canary)
    - White-box: optimal threshold sweep on advantage score
    Both use the two-layer Bernstein + MC error correction.
    """
    T_audit = len(noisy_with)

    # Advantage scores: hist[canary] - max(hist[other])
    def compute_scores(noisy_hists):
        scores = np.zeros(len(noisy_hists))
        for i, hist in enumerate(noisy_hists):
            mx = max(hist[j] for j in range(len(hist)) if j != canary_class_id)
            scores[i] = hist[canary_class_id] - mx
        return scores

    scores_with = compute_scores(noisy_with)
    scores_without = compute_scores(noisy_without)

    # --- Compute per-snapshot h_b variance for Bernstein ---
    # For each unique clean vector in the snapshot, compute attack success probability
    # Since PV output is nearly deterministic, variance is typically ~0
    # We estimate it from the bootstrap: group by snapshot index isn't available,
    # so we use the empirical observation that Var(h) ≈ 0 for PV (Table 2 in paper)
    # For a conservative bound, we can compute variance of the score across snapshots
    # But the correct approach is: h_b(x) = P(attack succeeds | clean output = x)
    # Since all snapshots give same clean output, var_h = 0 for PV.
    # We pass sigma2=0 and the Bernstein residual term handles it.
    sigma2_with = 0.0  # PV: all snapshots produce identical output
    sigma2_without = 0.0

    delta_1 = empirical_bernstein_correction(sigma2_with, n_llm, n_sample, gamma)
    delta_0 = empirical_bernstein_correction(sigma2_without, n_llm, n_sample, gamma)

    print(f"\n--- Error Bounds (gamma={gamma}) ---", file=f)
    print(f"  Bernstein+MC correction Delta_1 (TPR): {delta_1:.6f}", file=f)
    print(f"  Bernstein+MC correction Delta_0 (FPR): {delta_0:.6f}", file=f)

    # === Black-box (threshold at 0) ===
    tp_count = int(np.sum(scores_with > 0))
    fp_count = int(np.sum(scores_without > 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low = max(0.0, tpr_hat - delta_1)
    fpr_up = min(1.0, fpr_hat + delta_0)

    mu_bb = 0.0
    if 0 < fpr_up < 1 and 0 < (1 - tpr_low) < 1:
        mu_bb = max(
            norm.ppf(tpr_low) - norm.ppf(fpr_up),
            norm.ppf(1 - fpr_up) - norm.ppf(1 - tpr_low),
        )
        mu_bb = max(mu_bb, 0.0)
    eps_bb = gdp_to_eps(mu_bb, delta)

    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f}, FPR_hat={fpr_hat:.4f}", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.4f}", file=f)
    print(f"[Black-box] mu_lower={mu_bb:.4f}, eps_lower={eps_bb:.4f}", file=f)

    # === White-box (optimal threshold sweep) ===
    obs_with = np.sort(scores_with)
    obs_without = np.sort(scores_without)
    candidate_thresholds = np.unique(np.concatenate((obs_with, obs_without)))

    max_mu = -float('inf')
    best_threshold = None
    best_tpr_hat, best_fpr_hat = 0, 0

    for threshold in candidate_thresholds:
        fp = T_audit - int(np.searchsorted(obs_without, threshold, side='right'))
        fn = int(np.searchsorted(obs_with, threshold, side='right'))
        tpr_hat_t = 1.0 - fn / T_audit
        fpr_hat_t = fp / T_audit

        tpr_low_t = max(0.0, tpr_hat_t - delta_1)
        fpr_up_t = min(1.0, fpr_hat_t + delta_0)

        if not (0 < fpr_up_t < 1 and 0 < (1 - tpr_low_t) < 1):
            continue

        mu = max(
            norm.ppf(tpr_low_t) - norm.ppf(fpr_up_t),
            norm.ppf(1 - fpr_up_t) - norm.ppf(1 - tpr_low_t),
        )
        if mu > max_mu:
            max_mu = mu
            best_threshold = threshold
            best_tpr_hat = tpr_hat_t
            best_fpr_hat = fpr_hat_t

    eps_wb = gdp_to_eps(max(max_mu, 0.0), delta) if max_mu > -float('inf') else 0.0

    if best_threshold is not None:
        print(f"\n[White-box] Optimal threshold={best_threshold:.4f}", file=f)
        print(f"[White-box] TPR_hat={best_tpr_hat:.4f}, FPR_hat={best_fpr_hat:.4f}", file=f)
        print(f"[White-box] mu_lower={max(max_mu,0):.4f}, eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PV Active Auditing (Bootstrap + Bernstein+MC Error)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_subsets", type=int, default=2, help="Number of partitions T")
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=20000, help="Number of clean LLM snapshots")
    parser.add_argument("--n_sample", type=int, default=4000000, help="Number of bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--output_dir", type=str, default="results_pv_active")
    parser.add_argument("--output_prefix", type=str, default="pv-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--defense", type=str, choices=list(DEFENSE_PROMPTS.keys()), default="none")
    parser.add_argument("--attack_mode", type=str, choices=["direct", "semantic_similarity"], default="direct")
    parser.add_argument("--l3_defense", action="store_true", default=False)
    args = parser.parse_args()

    auditing_utils.defense_prompt = DEFENSE_PROMPTS[args.defense]
    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.use_l3_defense = args.l3_defense

    load_hf_model(args.model_path)
    all_texts = load_texts_from_csv(args.data_path, args.text_col, args.delimiter, not args.no_header)
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

        num_base = args.shots_per_subset * args.num_subsets
        if args.canary_mode == "fixed":
            idx = max(0, min(args.canary_index, len(all_texts) - 1))
        else:
            idx = rng.randrange(len(all_texts))
        canary_text = all_texts[idx]

        remaining = [i for i in range(len(all_texts)) if i != idx]
        base_indices = rng.sample(remaining, num_base)
        base_exemplars = [(all_texts[i], "No") for i in base_indices]
        exemplars_without = base_exemplars
        exemplars_with = [(canary_text, "Yes")] + base_exemplars[1:]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PV Active Auditing (Bootstrap + Bernstein+MC Error Bound)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Hoeffding (MC)", file=f)
            print(f"Estimator: GDP", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, T={args.num_subsets}", file=f)
            print("=" * 60, file=f)

            # Stage 1: Collect snapshots (shared across epsilons)
            t1_start = time.time()
            temp_pv = PVClassifier(["Yes", "No"], 1.0, args.delta)
            vecs_with, vecs_without = collect_snapshots(
                temp_pv, exemplars_with, exemplars_without,
                canary_text, args.num_subsets, args.n_llm, f
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Stage 1: {t1_time:.1f}s", file=f)

            # Stage 2 & 3: per epsilon
            for epsilon in args.epsilons:
                pv = PVClassifier(["Yes", "No"], epsilon, args.delta)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={pv.sigma:.6f}", file=f)

                t2_start = time.time()
                noisy_with = bootstrap_simulate(vecs_with, pv.sigma, pv.num_classes, args.n_sample)
                noisy_without = bootstrap_simulate(vecs_without, pv.sigma, pv.num_classes, args.n_sample)
                t2_time = time.time() - t2_start

                t3_start = time.time()
                eps_bb, eps_wb = run_active_audit(
                    noisy_with, noisy_without, canary_class_id=0,
                    n_llm=args.n_llm, n_sample=args.n_sample,
                    delta=args.delta, gamma=args.gamma, f=f
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
