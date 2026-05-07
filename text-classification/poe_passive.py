"""
PoE Passive Auditing (Full Process).

Runs the full Product-of-Experts pipeline end-to-end for each trial:
  1. Each exemplar is an independent expert (log-prob extraction + clipping)
  2. Sum clipped log-probs → utility vector
  3. Exponential mechanism selects output class

Uses Clopper-Pearson CI on observed FPR/FNR.
Uses pure-DP estimator (exponential mechanism → pure DP).

Usage:
    python poe_passive.py --data_path /path/to/data.csv --epsilons 1 2 4 8 16
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
    load_hf_model, load_texts_from_csv, build_poe_expert_prompt,
    get_label_token_ids, get_log_probs_for_labels,
    clopper_pearson, pure_dp_eps_lower_bound, DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PoE Mechanism
# ==============================================================================
def clip_log_probs(log_probs: np.ndarray, gamma: float) -> np.ndarray:
    """Clip log-probabilities to [-gamma, 0]. Values outside → 0 (abstain)."""
    result = log_probs.copy()
    result[(log_probs < -gamma) | (log_probs > 0.0)] = 0.0
    return result


class PoEClassifier:
    def __init__(self, epsilon, delta, gamma=2.0):
        self.class_labels = ["Yes", "No"]
        self.num_classes = 2
        self.label_to_id = {"Yes": 0, "No": 1}
        self.id_to_label = {0: "Yes", 1: "No"}
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.label_token_ids = None

    def init_label_tokens(self):
        self.label_token_ids = get_label_token_ids(self.class_labels)

    def get_clean_utility_vector(self, context_texts, query_text):
        """Sum of clipped log-probs across all experts."""
        utility = np.zeros(self.num_classes)
        for text in context_texts:
            prompt = build_poe_expert_prompt(text, query_text)
            log_probs = get_log_probs_for_labels(prompt, self.label_token_ids)
            utility += clip_log_probs(log_probs, self.gamma)
        return utility

    def run_full_process(self, context_texts, query_text):
        """Run full PoE: compute utility + exponential mechanism sampling."""
        utility = self.get_clean_utility_vector(context_texts, query_text)
        scaled = utility * self.epsilon / (2.0 * self.gamma)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= probs.sum()
        chosen = np.random.choice(self.num_classes, p=probs)
        return self.id_to_label[chosen], utility


# ==============================================================================
# Passive Audit
# ==============================================================================
def run_passive_audit(poe, contexts_with, contexts_without, query,
                      n_trials, delta, f):
    canary_class_id = 0  # "Yes"

    scores_with = []
    bb_tp = 0
    for _ in tqdm(range(n_trials), desc="Passive trials (with canary)"):
        label, utility = poe.run_full_process(contexts_with, query)
        if label == "Yes":
            bb_tp += 1
        # White-box: Gumbel-noised advantage = utility[Yes] - utility[No] + Gumbel difference
        gumbel_scale = 2.0 * poe.gamma / poe.epsilon
        noisy_util = utility + np.random.gumbel(0, gumbel_scale, poe.num_classes)
        mx = max(noisy_util[j] for j in range(poe.num_classes) if j != canary_class_id)
        scores_with.append(noisy_util[canary_class_id] - mx)

    scores_without = []
    bb_fp = 0
    for _ in tqdm(range(n_trials), desc="Passive trials (without canary)"):
        label, utility = poe.run_full_process(contexts_without, query)
        if label == "Yes":
            bb_fp += 1
        gumbel_scale = 2.0 * poe.gamma / poe.epsilon
        noisy_util = utility + np.random.gumbel(0, gumbel_scale, poe.num_classes)
        mx = max(noisy_util[j] for j in range(poe.num_classes) if j != canary_class_id)
        scores_without.append(noisy_util[canary_class_id] - mx)

    confidence = 0.95

    # === Black-box ===
    bb_fn = n_trials - bb_tp
    _, alpha_upper_bb = clopper_pearson(bb_fp, n_trials, confidence)
    _, beta_upper_bb = clopper_pearson(bb_fn, n_trials, confidence)
    eps_bb = pure_dp_eps_lower_bound(alpha_upper_bb, beta_upper_bb, delta)
    print(f"\n[Black-box] TPR={bb_tp/n_trials:.4f}, FPR={bb_fp/n_trials:.4f}", file=f)
    print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds) ===
    obs_with = np.sort(np.array(scores_with))
    obs_without = np.sort(np.array(scores_without))
    candidates = np.unique(np.concatenate((obs_with, obs_without)))

    max_eps = -float("inf")
    best_threshold = None
    best_fp, best_fn = -1, -1

    for threshold in candidates:
        fp = n_trials - int(np.searchsorted(obs_without, threshold, side="right"))
        fn = int(np.searchsorted(obs_with, threshold, side="right"))
        _, au = clopper_pearson(fp, n_trials, confidence)
        _, bu = clopper_pearson(fn, n_trials, confidence)
        if au >= 1 or bu >= 1 or au <= 0 or bu <= 0:
            continue
        cur_eps = pure_dp_eps_lower_bound(au, bu, delta)
        if cur_eps > max_eps:
            max_eps, best_threshold = cur_eps, threshold
            best_fp, best_fn = fp, fn

    eps_wb = max_eps if max_eps > 0 else 0.0
    if best_threshold is not None:
        print(f"[White-box] Optimal threshold={best_threshold:.4f}", file=f)
        print(f"[White-box] FPR={best_fp/n_trials:.4f}, FNR={best_fn/n_trials:.4f}", file=f)
        print(f"[White-box] eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PoE Passive Auditing (Full Process)")
    parser.add_argument("--model_path", type=str, default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF")
    parser.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/ag_news_csv/test.csv")
    parser.add_argument("--text_col", type=int, default=2)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_exemplars", type=int, default=8, help="J: number of experts")
    parser.add_argument("--n_trials", type=int, default=2000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=2.0, help="Log-prob clipping bound")
    parser.add_argument("--output_dir", type=str, default="results_poe_passive")
    parser.add_argument("--output_prefix", type=str, default="poe-passive")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, choices=["direct", "semantic_similarity"], default="semantic_similarity")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
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
        if args.canary_mode == "fixed":
            idx = max(0, min(args.canary_index, len(all_texts) - 1))
        else:
            idx = rng.randrange(len(all_texts))
        canary_text = all_texts[idx]

        remaining = [i for i in range(len(all_texts)) if i != idx]
        base_indices = rng.sample(remaining, args.num_exemplars)
        base_texts = [all_texts[i] for i in base_indices]
        contexts_without = base_texts
        contexts_with = [canary_text] + base_texts[1:]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PoE Passive Auditing (Full Process)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Mechanism: Exponential (pure DP)", file=f)
            print(f"Confidence: Clopper-Pearson 95%", file=f)
            print(f"Estimator: Pure DP", file=f)
            print(f"n_trials={args.n_trials}, J={args.num_exemplars}, gamma={args.gamma}", file=f)
            print("=" * 60, file=f)

            poe = PoEClassifier(1.0, args.delta, args.gamma)
            poe.init_label_tokens()

            for epsilon in args.epsilons:
                poe_eps = PoEClassifier(epsilon, args.delta, args.gamma)
                poe_eps.label_token_ids = poe.label_token_ids

                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

                t0 = time.time()
                eps_bb, eps_wb = run_passive_audit(
                    poe_eps, contexts_with, contexts_without, canary_text,
                    args.n_trials, args.delta, f
                )
                elapsed = time.time() - t0

                print(f"\n[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"[TIMING] {elapsed:.1f}s total", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
