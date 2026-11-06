"""
PV Passive Auditing (Full Process).

Runs the full DP-ICL Private Voting pipeline end-to-end for each trial.
Each trial makes real LLM calls, adds Gaussian noise, and records the outcome.
Uses Clopper-Pearson confidence intervals on the observed FPR/FNR.
Converts to epsilon via GDP estimator (Gaussian mechanism → GDP).

Usage:
    python pv_passive.py --data_path /path/to/data.csv --epsilons 1 2 4 8 16
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
    load_hf_model, load_texts_from_csv, llm_yes_no_batch,
    analytic_gauss_sigma, clopper_pearson, gdp_to_eps,
    DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PV Mechanism
# ==============================================================================
class PVClassifier:
    """Private Voting classifier with analytic Gaussian mechanism."""

    def __init__(self, class_labels, epsilon, delta, classical_noise=False, variance_as_std=False):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        sensitivity = math.sqrt(2)  # L2 sensitivity of vote histogram
        self.sigma = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon if classical_noise
                      else analytic_gauss_sigma(sensitivity, epsilon, delta))
        if variance_as_std:  # DP flaw: sigma^2 is passed where the sampler expects a standard deviation
            self.sigma = self.sigma ** 2

    def run_full_processes(self, exemplars, query, num_subsets, n_trials, batch_size, desc=None):
        """Run n_trials full DP-ICL processes: partition → vote → add noise → argmax.
        Every trial's partition is drawn first, all subset votes go through the LLM
        in batched passes, then each trial adds its own noise."""
        trial_subsets = []
        for _ in range(n_trials):
            shuffled = np.random.permutation(len(exemplars))
            trial_subsets.append([[exemplars[i][0] for i in s]
                                  for s in np.array_split(shuffled, num_subsets) if len(s) > 0])
        votes = llm_yes_no_batch([texts for subsets in trial_subsets for texts in subsets],
                                 query, batch_size, desc)

        results, pos = [], 0
        for subsets in trial_subsets:
            histogram = np.zeros(self.num_classes)
            for vote in votes[pos:pos + len(subsets)]:
                if vote in self.label_to_id:
                    histogram[self.label_to_id[vote]] += 1
            pos += len(subsets)
            noisy_histogram = histogram + np.random.normal(0, self.sigma, self.num_classes)
            winner = np.argmax(noisy_histogram)
            results.append((self.id_to_label[winner], noisy_histogram))
        return results


# ==============================================================================
# Passive Audit
# ==============================================================================
def run_passive_audit(pv, exemplars_with, exemplars_without, query,
                      canary_class_id, num_subsets, n_trials, delta, f, batch_size):
    """
    Run n_trials full-process DP-ICL calls for both with/without canary.
    Compute black-box and white-box epsilon lower bounds.
    """
    print(f"\n--- Running {n_trials} full-process passive trials ---", file=f)

    # Black-box: does argmax match canary class?
    # White-box: advantage score = noisy_hist[canary] - max(noisy_hist[other])
    scores_with = []
    bb_tp = 0
    for label, noisy_hist in pv.run_full_processes(exemplars_with, query, num_subsets, n_trials,
                                                   batch_size, desc="Passive trials (with canary)"):
        if label == "Yes":
            bb_tp += 1
        mx = max(noisy_hist[j] for j in range(len(noisy_hist)) if j != canary_class_id)
        scores_with.append(noisy_hist[canary_class_id] - mx)

    scores_without = []
    bb_fp = 0
    for label, noisy_hist in pv.run_full_processes(exemplars_without, query, num_subsets, n_trials,
                                                   batch_size, desc="Passive trials (without canary)"):
        if label == "Yes":
            bb_fp += 1
        mx = max(noisy_hist[j] for j in range(len(noisy_hist)) if j != canary_class_id)
        scores_without.append(noisy_hist[canary_class_id] - mx)

    confidence = 0.95

    # === Black-box result (threshold at argmax = canary class) ===
    bb_fn = n_trials - bb_tp
    _, alpha_upper_bb = clopper_pearson(bb_fp, n_trials, confidence)
    _, beta_upper_bb = clopper_pearson(bb_fn, n_trials, confidence)
    mu_bb = 0.0
    if 0 < alpha_upper_bb < 1 and 0 < beta_upper_bb < 1:
        mu_bb = max(
            norm.ppf(1 - beta_upper_bb) - norm.ppf(alpha_upper_bb),
            norm.ppf(1 - alpha_upper_bb) - norm.ppf(beta_upper_bb),
        )
    eps_bb = gdp_to_eps(mu_bb, delta)
    print(f"\n[Black-box] TPR={bb_tp/n_trials:.4f}, FPR={bb_fp/n_trials:.4f}", file=f)
    print(f"[Black-box] mu_lower={mu_bb:.4f}, eps_lower={eps_bb:.4f}", file=f)

    # === White-box result (optimal threshold on advantage score) ===
    obs_with = np.sort(np.array(scores_with))
    obs_without = np.sort(np.array(scores_without))
    candidate_thresholds = np.unique(np.concatenate((obs_with, obs_without)))

    max_mu = -float('inf')
    best_threshold = None
    best_fp, best_fn = -1, -1

    for threshold in candidate_thresholds:
        fp = n_trials - int(np.searchsorted(obs_without, threshold, side='right'))
        fn = int(np.searchsorted(obs_with, threshold, side='right'))
        _, au = clopper_pearson(fp, n_trials, confidence)
        _, bu = clopper_pearson(fn, n_trials, confidence)
        if not (0 < au < 1 and 0 < bu < 1):
            continue
        mu = max(
            norm.ppf(1 - bu) - norm.ppf(au),
            norm.ppf(1 - au) - norm.ppf(bu),
        )
        if mu > max_mu:
            max_mu = mu
            best_threshold = threshold
            best_fp, best_fn = fp, fn

    eps_wb = gdp_to_eps(max_mu, delta) if max_mu > 0 else 0.0
    if best_threshold is not None:
        print(f"[White-box] Optimal threshold={best_threshold:.4f}", file=f)
        print(f"[White-box] FPR={best_fp/n_trials:.4f}, FNR={best_fn/n_trials:.4f}", file=f)
        print(f"[White-box] mu_lower={max_mu:.4f}, eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PV Passive Auditing (Full Process)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_subsets", type=int, default=2, help="Number of partitions T")
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=20000, help="Number of full-process trials")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="results_pv_passive")
    parser.add_argument("--output_prefix", type=str, default="pv-passive")
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
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM generate call (1 = one call at a time)")
    args = parser.parse_args()

    # Set global config
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

        # Select canary and exemplars
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
            print("PV Passive Auditing (Full Process)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Confidence: Clopper-Pearson 95%", file=f)
            print(f"Estimator: GDP (Gaussian mechanism)", file=f)
            print(f"n_trials={args.n_trials}, T={args.num_subsets}", file=f)
            print("=" * 60, file=f)

            for epsilon in args.epsilons:
                pv = PVClassifier(["Yes", "No"], epsilon, args.delta, args.classical_noise, args.variance_as_std)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={pv.sigma:.6f}", file=f)

                t0 = time.time()
                eps_bb, eps_wb = run_passive_audit(
                    pv, exemplars_with, exemplars_without, canary_text,
                    canary_class_id=0, num_subsets=args.num_subsets,
                    n_trials=args.n_trials, delta=args.delta, f=f,
                    batch_size=args.batch_size
                )
                elapsed = time.time() - t0

                print(f"\n[RESULT] eps_theory={epsilon}: black-box={eps_bb:.4f}, white-box={eps_wb:.4f}", file=f)
                print(f"[TIMING] {elapsed:.1f}s total, {elapsed/args.n_trials*1000:.2f}ms per trial", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        wall = time.time() - t_run
        with open(output_file, "a") as f:
            print(f"\n[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons "
                  f"(LLM + audit, excl. model load)", file=f)
        print(f"[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
