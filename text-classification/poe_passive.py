"""
PoE Passive Auditing (Full Process).

Runs the full Product-of-Experts pipeline end-to-end for each trial:
  1. Each exemplar is an independent expert (log-prob extraction + clipping)
  2. Sum clipped log-probs → utility vector
  3. Exponential mechanism selects output class

Uses empirical-Bernstein (Maurer-Pontil) bounds on observed FPR/FNR (one DP draw per LLM output).
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
    get_label_token_ids, get_log_probs_for_labels_batch,
    empirical_bernstein_interval, pure_dp_eps_lower_bound, DEFENSE_PROMPTS,
)
import auditing_utils


# ==============================================================================
# PoE Mechanism
# ==============================================================================
def clip_log_probs(log_probs: np.ndarray, gamma: float, mode: str = "abstain") -> np.ndarray:
    """Clip log-probabilities to [-gamma, 0].
    abstain: values outside -> 0 (original PoE).
    clamp:   values outside -> nearest bound (same as poe_active.py --clip clamp)."""
    if mode == "clamp":
        return np.clip(log_probs, -gamma, 0.0)
    result = log_probs.copy()
    result[(log_probs < -gamma) | (log_probs > 0.0)] = 0.0
    return result


class PoEClassifier:
    def __init__(self, epsilon, delta, gamma=2.0, clip="abstain"):
        self.clip = clip
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

    def get_clean_utility_vectors(self, context_lists, query_text, batch_size, desc=None):
        """Sum of clipped log-probs across all experts, for every context list.
        All expert forward passes of all lists go through the LLM in batched passes."""
        prompts = [build_poe_expert_prompt(text, query_text)
                   for contexts in context_lists for text in contexts]
        log_probs = get_log_probs_for_labels_batch(prompts, self.label_token_ids, batch_size, desc)
        utilities, pos = [], 0
        for contexts in context_lists:
            utility = np.zeros(self.num_classes)
            for lp in log_probs[pos:pos + len(contexts)]:
                utility += clip_log_probs(lp.astype(float), self.gamma, self.clip)
            utilities.append(utility)
            pos += len(contexts)
        return utilities

    def run_full_process(self, utility):
        """Run full PoE given the (batched) utility: exponential mechanism sampling."""
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
                      n_trials, delta, f, batch_size, confidence=0.95):
    canary_class_id = 0  # "Yes"

    # Every trial runs its own expert forward passes; they are batched up front,
    # then the EM + Gumbel draws run per trial in the original np.random order.
    utils_with = poe.get_clean_utility_vectors(
        [contexts_with] * n_trials, query, batch_size, desc="Passive LLM (with canary)")
    utils_without = poe.get_clean_utility_vectors(
        [contexts_without] * n_trials, query, batch_size, desc="Passive LLM (without canary)")

    scores_with = []
    bb_tp = 0
    for utility in tqdm(utils_with, desc="Passive trials (with canary)"):
        label, utility = poe.run_full_process(utility)
        if label == "Yes":
            bb_tp += 1
        # White-box: Gumbel-noised advantage = utility[Yes] - utility[No] + Gumbel difference
        gumbel_scale = 2.0 * poe.gamma / poe.epsilon
        noisy_util = utility + np.random.gumbel(0, gumbel_scale, poe.num_classes)
        mx = max(noisy_util[j] for j in range(poe.num_classes) if j != canary_class_id)
        scores_with.append(noisy_util[canary_class_id] - mx)

    scores_without = []
    bb_fp = 0
    for utility in tqdm(utils_without, desc="Passive trials (without canary)"):
        label, utility = poe.run_full_process(utility)
        if label == "Yes":
            bb_fp += 1
        gumbel_scale = 2.0 * poe.gamma / poe.epsilon
        noisy_util = utility + np.random.gumbel(0, gumbel_scale, poe.num_classes)
        mx = max(noisy_util[j] for j in range(poe.num_classes) if j != canary_class_id)
        scores_without.append(noisy_util[canary_class_id] - mx)

    # === Black-box ===
    bb_fn = n_trials - bb_tp
    _, alpha_upper_bb = empirical_bernstein_interval(bb_fp, n_trials, confidence)
    _, beta_upper_bb = empirical_bernstein_interval(bb_fn, n_trials, confidence)
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
        _, au = empirical_bernstein_interval(fp, n_trials, confidence)
        _, bu = empirical_bernstein_interval(fn, n_trials, confidence)
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
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=1)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_exemplars", type=int, nargs="+", default=[1],
                        help="J values (number of experts) to audit; with several J, each uses "
                             "a Bonferroni-split confidence 1-0.05/len(J) and the best over J is reported")
    parser.add_argument("--n_trials", type=int, default=10000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=4.5, help="Log-prob clipping bound")
    parser.add_argument("--clip", choices=["abstain", "clamp"], default="abstain",
                        help="Out-of-range log-probs: abstain (-> 0) or clamp (-> nearest bound)")
    parser.add_argument("--output_dir", type=str, default="results_poe_passive")
    parser.add_argument("--output_prefix", type=str, default="poe-passive")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, choices=["direct", "semantic_similarity"], default="direct")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Logit temperature (<1.0 sharpens distribution)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM forward pass")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.logit_temperature = args.temperature
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

        # One fixed draw W of j_max exemplars per run, shared across J (as in poe_active.py):
        #   without_J = W[:J],  with_J = [canary] + W[:J-1]  (W[J-1] swapped out)
        js = args.num_exemplars
        confidence = 1.0 - 0.05 / len(js)
        remaining = [i for i in range(len(all_texts)) if i != idx]
        base_indices = rng.sample(remaining, max(js))
        base_texts = [all_texts[i] for i in base_indices]
        contexts = {J: ([canary_text] + base_texts[:J - 1], base_texts[:J]) for J in js}

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        t_run = time.time()
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PoE Passive Auditing (Full Process)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Mechanism: Exponential (pure DP)", file=f)
            if len(js) == 1:
                print(f"Confidence: empirical-Bernstein 95%", file=f)
            else:
                print(f"Confidence: empirical-Bernstein {confidence:.4g} per J "
                      f"(Bonferroni over {len(js)} J)", file=f)
            print(f"Estimator: Pure DP", file=f)
            jstr = js[0] if len(js) == 1 else js
            print(f"n_trials={args.n_trials}, J={jstr}, gamma={args.gamma}", file=f)
            print(f"Clip: {args.clip}, logit temperature={args.temperature}", file=f)
            print("=" * 60, file=f)

            poe = PoEClassifier(1.0, args.delta, args.gamma, args.clip)
            poe.init_label_tokens()

            for epsilon in args.epsilons:
                poe_eps = PoEClassifier(epsilon, args.delta, args.gamma, args.clip)
                poe_eps.label_token_ids = poe.label_token_ids

                if len(js) == 1:
                    contexts_with, contexts_without = contexts[js[0]]
                    print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

                    t0 = time.time()
                    eps_bb, eps_wb = run_passive_audit(
                        poe_eps, contexts_with, contexts_without, canary_text,
                        args.n_trials, args.delta, f, args.batch_size, confidence
                    )
                    elapsed = time.time() - t0

                    print(f"\n[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                    print(f"[TIMING] {elapsed:.1f}s total", file=f)
                    print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")
                    continue

                best_bb, best_wb = (0.0, None), (0.0, None)
                for J in js:
                    contexts_with, contexts_without = contexts[J]
                    print(f"\n{'='*40} eps_theory={epsilon}, J={J} {'='*40}", file=f)

                    t0 = time.time()
                    eps_bb, eps_wb = run_passive_audit(
                        poe_eps, contexts_with, contexts_without, canary_text,
                        args.n_trials, args.delta, f, args.batch_size, confidence
                    )
                    elapsed = time.time() - t0

                    print(f"\n[RESULT] eps_theory={epsilon}, J={J}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                    print(f"[TIMING] {elapsed:.1f}s total", file=f)
                    if best_bb[1] is None or eps_bb > best_bb[0]:
                        best_bb = (eps_bb, J)
                    if best_wb[1] is None or eps_wb > best_wb[0]:
                        best_wb = (eps_wb, J)

                print(f"\n[BEST] eps_theory={epsilon}: bb={best_bb[0]:.4f} (J={best_bb[1]}), "
                      f"wb={best_wb[0]:.4f} (J={best_wb[1]}) "
                      f"[Bonferroni over {len(js)} J, confidence={confidence:.4g}]", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={best_bb[0]:.4f} (J={best_bb[1]}), "
                      f"wb={best_wb[0]:.4f} (J={best_wb[1]})")

        wall = time.time() - t_run
        with open(output_file, "a") as f:
            print(f"\n[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons "
                  f"(LLM + audit, excl. model load)", file=f)
        print(f"[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
