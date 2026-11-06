"""
DP Flaw: PV with Classical Gaussian Noise.

Demonstrates that the classical Gaussian calibration sigma = Delta * sqrt(2*ln(1.25/delta)) / eps
is insufficient at high epsilon, leading to privacy violations.
The analytic Gaussian (Balle & Wang 2018) provides the correct tight calibration.

This script runs active auditing (bootstrap) with the CLASSICAL (loose) sigma
to show empirical epsilon exceeding theoretical epsilon at large eps_theory.

Usage:
    python pv_flaw_classical_gaussian.py --epsilons 1 2 4 8 16
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
    clopper_pearson, gdp_to_eps,
    DEFENSE_PROMPTS,
)
import auditing_utils


class PVClassifierClassical:
    """PV with CLASSICAL (loose) Gaussian noise calibration."""

    def __init__(self, class_labels, epsilon, delta):
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        sensitivity = math.sqrt(2)
        # Classical (loose) formula - insufficient for large epsilon
        self.sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    def get_clean_vote_vector(self, exemplars, query, num_subsets):
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


def main():
    parser = argparse.ArgumentParser(description="DP Flaw: PV Classical Gaussian")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--num_subsets", type=int, default=1)
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=20000)
    parser.add_argument("--n_sample", type=int, default=10000000, help="10M bootstrap for precise flaw detection")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--snapshot_gamma_frac", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results_pv_flaw")
    parser.add_argument("--output_prefix", type=str, default="pv-flaw-classical")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--attack_mode", type=str, default="direct")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.generation_temperature = args.temperature
    load_hf_model(args.model_path)
    all_texts = load_texts_from_csv(args.data_path, args.text_col)
    if not all_texts:
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
        canary_text = all_texts[idx]

        num_base = args.shots_per_subset * args.num_subsets
        remaining = [i for i in range(len(all_texts)) if i != idx]
        base_indices = rng.sample(remaining, num_base)
        base_exemplars = [(all_texts[i], "No") for i in base_indices]
        exemplars_without = base_exemplars
        exemplars_with = [(canary_text, "Yes")] + base_exemplars[1:]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("DP FLAW: PV with Classical (Loose) Gaussian Noise", file=f)
            print("Expected: eps_empirical > eps_theory at large eps", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print("=" * 60, file=f)

            # Stage 1: collect snapshots
            temp = PVClassifierClassical(["Yes", "No"], 1.0, args.delta)
            vecs_with, vecs_without = [], []
            for _ in tqdm(range(args.n_llm), desc="Snapshots (with)"):
                vecs_with.append(temp.get_clean_vote_vector(exemplars_with, canary_text, args.num_subsets))
            for _ in tqdm(range(args.n_llm), desc="Snapshots (without)"):
                vecs_without.append(temp.get_clean_vote_vector(exemplars_without, canary_text, args.num_subsets))

            print(f"Mean (with): {np.mean(vecs_with, axis=0)}", file=f)
            print(f"Mean (without): {np.mean(vecs_without, axis=0)}", file=f)

            for epsilon in args.epsilons:
                pv = PVClassifierClassical(["Yes", "No"], epsilon, args.delta)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma_classical={pv.sigma:.6f} (LOOSE)", file=f)

                # Bootstrap
                vecs_w = np.array(vecs_with)
                vecs_wo = np.array(vecs_without)
                idx_w = np.random.choice(len(vecs_w), args.n_sample, replace=True)
                idx_wo = np.random.choice(len(vecs_wo), args.n_sample, replace=True)
                noisy_w = vecs_w[idx_w] + np.random.normal(0, pv.sigma, (args.n_sample, pv.num_classes))
                noisy_wo = vecs_wo[idx_wo] + np.random.normal(0, pv.sigma, (args.n_sample, pv.num_classes))

                # Black-box: threshold at 0
                canary_id = 0
                scores_w = noisy_w[:, canary_id] - np.max(np.delete(noisy_w, canary_id, axis=1), axis=1)
                scores_wo = noisy_wo[:, canary_id] - np.max(np.delete(noisy_wo, canary_id, axis=1), axis=1)

                tp_count = int(np.sum(scores_w > 0))
                fp_count = int(np.sum(scores_wo > 0))
                tpr_hat = tp_count / args.n_sample
                fpr_hat = fp_count / args.n_sample

                # Bernstein snapshot correction + Clopper-Pearson for MC
                gamma_snap = args.gamma * args.snapshot_gamma_frac
                gamma_mc = args.gamma * (1.0 - args.snapshot_gamma_frac)
                log_snap = math.log(2.0 / gamma_snap)
                bern_corr = (math.sqrt(2.0 * 0.0 * log_snap / args.n_llm)
                             + 7.0 * log_snap / (3.0 * (args.n_llm - 1)))
                cp_confidence = 1.0 - gamma_mc
                cp_tpr_lo, _ = clopper_pearson(tp_count, args.n_sample, cp_confidence)
                _, cp_fpr_hi = clopper_pearson(fp_count, args.n_sample, cp_confidence)
                tpr_low = max(0.0, cp_tpr_lo - bern_corr)
                fpr_up = min(1.0, cp_fpr_hi + bern_corr)

                mu_bb = 0.0
                if 0 < fpr_up < 1 and 0 < (1 - tpr_low) < 1:
                    mu_bb = max(norm.ppf(tpr_low) - norm.ppf(fpr_up), 0)
                eps_bb = gdp_to_eps(mu_bb, args.delta)

                violation = eps_bb > epsilon
                print(f"[Black-box] TPR={tpr_hat:.4f}, FPR={fpr_hat:.4f}", file=f)
                print(f"[Black-box] eps_empirical={eps_bb:.4f} vs eps_theory={epsilon}", file=f)
                print(f"[Black-box] VIOLATION={'YES' if violation else 'no'}", file=f)
                print(f"Run {run_id}, eps={epsilon}: eps_emp={eps_bb:.4f} {'*** VIOLATION ***' if violation else ''}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
