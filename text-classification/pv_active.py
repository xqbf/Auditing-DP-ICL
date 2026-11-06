"""
PV Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean vote vectors via real LLM calls (snapshot).
Stage 2: Bootstrap n_sample noisy histograms by resampling + Gaussian noise.
Stage 3: Compute TPR/FPR with Bernstein (snapshot) + Clopper-Pearson (MC),
         then convert to epsilon via GDP estimator.

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
from scipy.special import ndtr
from scipy.stats import norm

from auditing_utils import (
    load_hf_model, load_texts_from_csv, llm_yes_no,
    analytic_gauss_sigma, clopper_pearson, gdp_to_eps,
    DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PV Mechanism (clean vote collection)
# ==============================================================================
class PVClassifier:
    def __init__(self, class_labels, epsilon, delta, classical_noise=False, variance_as_std=False):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        sensitivity = math.sqrt(2)
        self.sigma = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon if classical_noise
                      else analytic_gauss_sigma(sensitivity, epsilon, delta))
        if variance_as_std:  # DP flaw: sigma^2 is passed where the sampler expects a standard deviation
            self.sigma = self.sigma ** 2

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
                     delta, gamma, f, snapshot_gamma_frac=0.5,
                     apply_snapshot_correction=True, clean_diffs=None, noise_sd=None):
    """
    Compute epsilon lower bound using:
    - Black-box: fixed threshold at 0 (argmax = canary)
    - White-box: optimal threshold sweep on advantage score
    Both use Bernstein (snapshot) + Clopper-Pearson (MC) error correction.

    apply_snapshot_correction: if False, the snapshot-phase Bernstein *variance*
        term is dropped (the clean snapshots are deterministic -- empirical
        variance is exactly 0 -- so sqrt(2*sigma2*log/n) vanishes). We still keep
        the empirical-Bernstein *range* term 7*log(2/gamma_snap)/(3*(n_llm-1)):
        observing 0 deviations over n_llm draws does not prove the deviation
        probability is exactly 0 (LLM inference is not bitwise-deterministic on
        GPU), so this rule-of-three term is the honest price of "0 observed
        failures != 0 probability" and should not be zeroed out.
        If True, the full Bernstein bound (variance + range) is applied.
    clean_diffs: optional (with, without) arrays of clean hist[canary] - hist[other],
        one per snapshot. When given, the Bernstein variance is the sample variance
        of h(x; t) = P(d(x) + N(0, noise_sd^2) > t) at each threshold t, instead of 0.
        Needed at temperature > 0, where snapshots differ. Assumes two classes.
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

    # --- Bernstein snapshot correction (no MC Hoeffding -- replaced by Clopper-Pearson) ---
    sigma2_with = 0.0  # PV: all snapshots produce identical output
    sigma2_without = 0.0

    gamma_snap = gamma * snapshot_gamma_frac
    gamma_mc = gamma * (1.0 - snapshot_gamma_frac)
    log_snap = math.log(2.0 / gamma_snap)

    range_term = 7.0 * log_snap / (3.0 * (n_llm - 1))
    if apply_snapshot_correction:
        # Full empirical Bernstein: variance term + range term.
        bern_with = math.sqrt(2.0 * sigma2_with * log_snap / n_llm) + range_term
        bern_without = math.sqrt(2.0 * sigma2_without * log_snap / n_llm) + range_term
    else:
        # Deterministic snapshots: variance term is 0, but keep the range term
        # (rule-of-three: 0 observed deviations over n_llm does not prove the
        # deviation probability is 0). gamma stays split -- gamma_snap pays for
        # this range term, gamma_mc pays for Clopper-Pearson.
        bern_with = range_term
        bern_without = range_term

    cp_confidence = 1.0 - gamma_mc

    if clean_diffs is not None:
        vals_with, cnt_with = np.unique(clean_diffs[0], return_counts=True)
        vals_without, cnt_without = np.unique(clean_diffs[1], return_counts=True)

    def snapshot_term(vals, cnts, threshold):
        # Empirical-Bernstein deviation for the n_llm per-snapshot rates h(x; t).
        h = ndtr((vals - threshold) / noise_sd)
        mean = float(cnts @ h) / n_llm
        sigma2 = float(cnts @ (h - mean) ** 2) / (n_llm - 1)
        return math.sqrt(2.0 * sigma2 * log_snap / n_llm) + range_term

    print(f"\n--- Error Bounds (gamma={gamma}, snapshot_gamma_frac={snapshot_gamma_frac}) ---", file=f)
    print(f"  Bernstein_1 (snapshot): {bern_with:.6f}", file=f)
    print(f"  Bernstein_0 (snapshot): {bern_without:.6f}", file=f)
    print(f"  Clopper-Pearson confidence: {cp_confidence}", file=f)
    if clean_diffs is not None:
        print(f"  Empirical variance at t=0: Bernstein_1={snapshot_term(vals_with, cnt_with, 0.0):.6f}, "
              f"Bernstein_0={snapshot_term(vals_without, cnt_without, 0.0):.6f}", file=f)

    # Helper: compute TPR lower / FPR upper using CP + Bernstein
    def get_tpr_low_fpr_up(tp_count, fp_count, n, threshold=0.0):
        cp_tpr_lo, _ = clopper_pearson(tp_count, n, cp_confidence)
        _, cp_fpr_hi = clopper_pearson(fp_count, n, cp_confidence)
        if clean_diffs is not None:
            tpr_low = max(0.0, cp_tpr_lo - snapshot_term(vals_with, cnt_with, threshold))
            fpr_up = min(1.0, cp_fpr_hi + snapshot_term(vals_without, cnt_without, threshold))
        else:
            tpr_low = max(0.0, cp_tpr_lo - bern_with)
            fpr_up = min(1.0, cp_fpr_hi + bern_without)
        return tpr_low, fpr_up

    # === Black-box (threshold at 0) ===
    tp_count = int(np.sum(scores_with > 0))
    fp_count = int(np.sum(scores_without > 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low, fpr_up = get_tpr_low_fpr_up(tp_count, fp_count, T_audit)

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
        tp = T_audit - fn

        tpr_low_t, fpr_up_t = get_tpr_low_fpr_up(tp, fp, T_audit, threshold)

        if not (0 < fpr_up_t < 1 and 0 < (1 - tpr_low_t) < 1):
            continue

        mu = max(
            norm.ppf(tpr_low_t) - norm.ppf(fpr_up_t),
            norm.ppf(1 - fpr_up_t) - norm.ppf(1 - tpr_low_t),
        )
        if mu > max_mu:
            max_mu = mu
            best_threshold = threshold
            best_tpr_hat = 1.0 - fn / T_audit
            best_fpr_hat = fp / T_audit

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
    parser = argparse.ArgumentParser(description="PV Active Auditing (Bootstrap + Bernstein+CP Error)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_subsets", type=int, default=2, help="Number of partitions T")
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=2000, help="Number of clean LLM snapshots")
    parser.add_argument("--n_sample", type=int, default=400000, help="Number of bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--snapshot_gamma_frac", type=float, default=0.5,
                        help="Fraction of gamma allocated to snapshot phase (>0.5 = tighter snapshot bound)")
    parser.add_argument("--snapshot_correction", choices=["auto", "bernstein", "none"], default="auto",
                        help="Snapshot-phase error correction. 'auto': skip it when the clean "
                             "snapshots are deterministic (zero variance), else use Bernstein. "
                             "'bernstein': always apply. 'none': always skip.")
    parser.add_argument("--empirical_variance", action="store_true",
                        help="Use the real per-threshold snapshot variance in the Bernstein bound "
                             "instead of 0 (required at temperature > 0)")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding")
    parser.add_argument("--output_dir", type=str, default="results_pv_active")
    parser.add_argument("--output_prefix", type=str, default="pv-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--variance_as_std", action="store_true",
                        help="DP flaw: sample noise with std sigma^2 instead of sigma")
    parser.add_argument("--classical_noise", action="store_true",
                        help="DP flaw: classical sigma = Delta*sqrt(2 ln(1.25/delta))/eps instead of analytic")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--defense", type=str, choices=list(DEFENSE_PROMPTS.keys()), default="none")
    parser.add_argument("--attack_mode", type=str, choices=["direct", "semantic_similarity"], default="direct")
    parser.add_argument("--l3_defense", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    args = parser.parse_args()

    auditing_utils.defense_prompt = DEFENSE_PROMPTS[args.defense]
    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.use_l3_defense = args.l3_defense
    auditing_utils.generation_temperature = args.temperature

    load_hf_model(args.model_path)
    all_texts = load_texts_from_csv(args.data_path, args.text_col, args.delimiter,
                                    not args.no_header, args.encoding)
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
        t_run = time.time()
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PV Active Auditing (Bootstrap + Bernstein+CP Error Bound)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Clopper-Pearson (MC)", file=f)
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

            # Resolve snapshot correction: deterministic snapshots (zero variance
            # across all n_llm vote vectors) carry no estimation error, so the
            # additive Bernstein term is skipped -- otherwise it collapses the
            # black-box tail bound at large epsilon.
            det_with = float(np.var(np.array(vecs_with), axis=0).max()) == 0.0
            det_without = float(np.var(np.array(vecs_without), axis=0).max()) == 0.0
            deterministic = det_with and det_without
            if args.snapshot_correction == "bernstein":
                apply_snapshot_correction = True
            elif args.snapshot_correction == "none":
                apply_snapshot_correction = False
            else:  # auto
                apply_snapshot_correction = not deterministic
            print(f"[SNAPSHOT] deterministic={deterministic}, "
                  f"apply_snapshot_correction={apply_snapshot_correction} "
                  f"(mode={args.snapshot_correction})", file=f)
            print(f"[SNAPSHOT] empirical_variance={args.empirical_variance}", file=f)
            clean_diffs = None
            if args.empirical_variance:
                arr_with, arr_without = np.array(vecs_with), np.array(vecs_without)
                clean_diffs = (arr_with[:, 0] - arr_with[:, 1], arr_without[:, 0] - arr_without[:, 1])

            # Stage 2 & 3: per epsilon
            for epsilon in args.epsilons:
                pv = PVClassifier(["Yes", "No"], epsilon, args.delta, args.classical_noise, args.variance_as_std)
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
                    delta=args.delta, gamma=args.gamma, f=f,
                    snapshot_gamma_frac=args.snapshot_gamma_frac,
                    apply_snapshot_correction=apply_snapshot_correction,
                    clean_diffs=clean_diffs, noise_sd=math.sqrt(2.0) * pv.sigma
                )
                t3_time = time.time() - t3_start

                total = t1_time + t2_time + t3_time
                print(f"\n[TIMING] Stage 2: {t2_time:.1f}s, Stage 3: {t3_time:.1f}s", file=f)
                print(f"[TIMING] Total: {total:.1f}s, per-sample: {total/args.n_sample*1000:.4f}ms", file=f)
                print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        wall = time.time() - t_run
        with open(output_file, "a") as f:
            print(f"\n[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons "
                  f"(LLM + audit, excl. model load)", file=f)
        print(f"[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
