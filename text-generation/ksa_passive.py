"""
KSA Passive Auditing (Full Process).

Runs the full KSA pipeline for each trial:
  1. Each subset generates Yes/No → vote histogram
  2. Joint Exponential Mechanism selects output (K=1 → standard EM)

Uses empirical-Bernstein (Maurer-Pontil) bounds. Uses pure-DP estimator (exponential mechanism).

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

from auditing_utils import (
    load_hf_model, load_texts, build_membership_prompt, generate_responses,
    parse_yes_no, empirical_bernstein_interval, pure_dp_eps_lower_bound,
)
import auditing_utils


def collect_vote_histograms(context_lists, query_text, batch_size, desc=None):
    """Query the LLM for every context of every snapshot in batched passes.
    Returns one [yes_count, no_count] histogram per entry of context_lists.
    Under greedy decoding each distinct prompt is sent to the LLM only once."""
    prompts = [build_membership_prompt(text, query_text)
               for contexts in context_lists for text in contexts]
    if auditing_utils.generation_temperature and auditing_utils.generation_temperature > 0:
        responses = generate_responses(prompts, batch_size, desc)
    else:
        uniq = list(dict.fromkeys(prompts))
        cached = dict(zip(uniq, generate_responses(uniq, batch_size, desc)))
        responses = [cached[p] for p in prompts]
    votes = [parse_yes_no(r) == "Yes" for r in responses]
    hists, pos = [], 0
    for contexts in context_lists:
        yes = sum(votes[pos:pos + len(contexts)])
        hists.append(np.array([yes, len(contexts) - yes], dtype=float))
        pos += len(contexts)
    return hists


def joint_em_select(histogram, epsilon):
    """Standard exponential mechanism (K=1). Returns 0=Yes, 1=No."""
    sensitivity = 1.0
    scores = histogram * epsilon / (2.0 * sensitivity)
    scores -= np.max(scores)
    probs = np.exp(scores)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def run_passive_audit(all_texts, canary_text, canary_index, num_exemplars,
                      n_trials, epsilons, delta, rng, f, batch_size, fixed_context=False):
    """num_exemplars: list of J values. A single J keeps the original behaviour.
    Several J (requires fixed_context) run a J sweep as in ksa_active.py: one fixed draw W
    of max(J) exemplars per run, without_J = W[:J], with_J = [canary] + W[:J-1], each J at
    empirical-Bernstein confidence 1-0.05/len(J) (Bonferroni); the best over J is reported."""
    js = list(num_exemplars)
    if len(js) > 1 and not fixed_context:
        raise ValueError("A J sweep (several --num_exemplars) requires --fixed_context")
    confidence = 1.0 - 0.05 / len(js)
    remaining = [i for i in range(len(all_texts)) if i != canary_index]
    # fixed_context: one context per run, shared by all epsilons and trials
    # (single J: with-canary world swaps the first exemplar, as in poe_passive.py).
    fixed_base = ([all_texts[i] for i in rng.sample(remaining, max(js))]
                  if fixed_context else None)

    for epsilon in epsilons:
        if len(js) == 1:
            num_exemplars = js[0]
            print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)

            # Draw all trial contexts first (same rng order), batch the LLM, then run
            # the EM + Gumbel draws per trial in the original np.random order.
            if fixed_context:
                contexts_with = [[canary_text] + fixed_base[1:]] * n_trials
                contexts_without = [fixed_base] * n_trials
            else:
                contexts_with = [[canary_text] + [all_texts[i] for i in rng.sample(remaining, num_exemplars - 1)]
                                 for _ in range(n_trials)]
                contexts_without = [[all_texts[i] for i in rng.sample(remaining, num_exemplars)]
                                    for _ in range(n_trials)]
            eps_bb, eps_wb = audit_one(contexts_with, contexts_without, canary_text, epsilon,
                                       n_trials, delta, confidence, f, batch_size)
            print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
            print(f"eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")
            continue

        best_bb, best_wb = (0.0, None), (0.0, None)
        for J in js:
            print(f"\n{'='*40} eps_theory={epsilon}, J={J} {'='*40}", file=f)
            contexts_with = [[canary_text] + fixed_base[:J - 1]] * n_trials
            contexts_without = [fixed_base[:J]] * n_trials
            eps_bb, eps_wb = audit_one(contexts_with, contexts_without, canary_text, epsilon,
                                       n_trials, delta, confidence, f, batch_size)
            print(f"[RESULT] eps_theory={epsilon}, J={J}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
            if best_bb[1] is None or eps_bb > best_bb[0]:
                best_bb = (eps_bb, J)
            if best_wb[1] is None or eps_wb > best_wb[0]:
                best_wb = (eps_wb, J)
        print(f"\n[BEST] eps_theory={epsilon}: bb={best_bb[0]:.4f} (J={best_bb[1]}), "
              f"wb={best_wb[0]:.4f} (J={best_wb[1]}) "
              f"[Bonferroni over {len(js)} J, confidence={confidence:.4g}]", file=f)
        print(f"eps={epsilon}: bb={best_bb[0]:.4f} (J={best_bb[1]}), "
              f"wb={best_wb[0]:.4f} (J={best_wb[1]})")


def audit_one(contexts_with, contexts_without, canary_text, epsilon,
              n_trials, delta, confidence, f, batch_size):
    """n_trials full processes per world for one (context pair, epsilon); returns (eps_bb, eps_wb)."""
    bb_tp, bb_fp = 0, 0
    scores_with, scores_without = [], []
    hists_with = collect_vote_histograms(
        contexts_with, canary_text, batch_size, desc=f"Passive (with, eps={epsilon})")
    hists_without = collect_vote_histograms(
        contexts_without, canary_text, batch_size, desc=f"Passive (without, eps={epsilon})")

    for hist in hists_with:
        # Full process: EM selection
        sel = joint_em_select(hist, epsilon)
        if sel == 0:  # "Yes"
            bb_tp += 1
        # White-box: Gumbel-noised advantage
        noisy = hist * epsilon / 2.0 + np.random.gumbel(0, 1.0, 2)
        scores_with.append(noisy[0] - noisy[1])

    for hist in hists_without:
        sel = joint_em_select(hist, epsilon)
        if sel == 0:
            bb_fp += 1
        noisy = hist * epsilon / 2.0 + np.random.gumbel(0, 1.0, 2)
        scores_without.append(noisy[0] - noisy[1])

    # Black-box
    bb_fn = n_trials - bb_tp
    _, au_bb = empirical_bernstein_interval(bb_fp, n_trials, confidence)
    _, bu_bb = empirical_bernstein_interval(bb_fn, n_trials, confidence)
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
        _, au = empirical_bernstein_interval(fp, n_trials, confidence)
        _, bu = empirical_bernstein_interval(fn, n_trials, confidence)
        if au >= 1 or bu >= 1 or au <= 0 or bu <= 0:
            continue
        cur = pure_dp_eps_lower_bound(au, bu, delta)
        if cur > max_eps:
            max_eps, best_threshold = cur, threshold

    eps_wb = max(max_eps, 0.0)
    print(f"[White-box] eps_lower={eps_wb:.4f}", file=f)
    return eps_bb, eps_wb


def main():
    parser = argparse.ArgumentParser(description="KSA Passive Auditing (Full Process)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_exemplars", type=int, nargs="+", default=[8],
                        help="J values; several run a J sweep (needs --fixed_context) with a "
                             "Bonferroni-split confidence 1-0.05/len(J)")
    parser.add_argument("--n_trials", type=int, default=50000)
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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM generate call")
    parser.add_argument("--fixed_context", action="store_true",
                        help="Draw the context once per run instead of once per trial")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.max_new_tokens_global = args.max_new_tokens
    auditing_utils.generation_temperature = args.temperature
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
        t_run = time.time()
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("KSA Passive Auditing (Full Process)", file=f)
            print(f"Estimator: Pure DP (exponential mechanism)", file=f)
            js = args.num_exemplars
            if len(js) == 1:
                print(f"Confidence: empirical-Bernstein 95%", file=f)
            else:
                print(f"Confidence: empirical-Bernstein {1.0 - 0.05 / len(js):.4g} per J "
                      f"(Bonferroni over {len(js)} J)", file=f)
            print(f"n_trials={args.n_trials}, J={js[0] if len(js) == 1 else js}", file=f)
            print(f"fixed_context={args.fixed_context}", file=f)
            print("=" * 60, file=f)

            run_passive_audit(all_texts, canary, idx, args.num_exemplars,
                              args.n_trials, args.epsilons, args.delta, rng, f, args.batch_size,
                              fixed_context=args.fixed_context)
        wall = time.time() - t_run
        with open(output_file, "a") as f:
            print(f"\n[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons "
                  f"(LLM + audit, excl. model load)", file=f)
        print(f"[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
