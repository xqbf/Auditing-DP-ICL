"""
KSA Active Auditing (honest two-phase: per-snapshot noise averaging + empirical Bernstein).

Phase 1 (expensive, LLM): collect n_llm clean vote histograms, each over an
    independently sampled exemplar context, so the audit reflects the genuine
    randomness of the DP-ICL input distribution (NOT a fixed context).
Phase 2 (cheap, noise): for each epsilon, draw m = n_sample / n_llm fresh
    Gumbel-max samples PER snapshot and average them into a per-snapshot canary-
    selection rate ghat_i in [0,1] (and per-snapshot advantage scores for the
    white-box sweep). This is the correct use of the noise budget: it drives down
    the within-snapshot noise variance E_X[g(1-g)], NOT a bootstrap resample that
    would over-count n_sample as if it were the effective sample size.
Phase 3 (estimate): bound TPR/FPR with empirical Bernstein (Maurer-Pontil) on the
    n_llm iid per-snapshot rates ghat_i, using their REAL empirical variance, then
    convert to epsilon via the pure-DP estimator.

Why the noise budget (n_sample) helps: by the law of total variance,
    TPR(1-TPR) = E_X[g(1-g)]  +  Var_X(g).
Averaging m noise draws per snapshot removes the first term as m grows, shrinking
the estimator variance from TPR(1-TPR)/n_llm toward Var_X(g)/n_llm. It pays off in
the bulk (moderate eps). It does NOT help the deep tail: as g->0 both terms vanish
and only the empirical-Bernstein range term (~1/n_llm) remains, so the recoverable
epsilon is floored by n_llm:
    ceiling ~= log((1-r)/r),  r = 7*log(2/gamma)/(3*(n_llm-1)).
  n_llm =  2,000 -> ~5.4      n_llm = 20,000 -> ~7.8
  n_llm = 10,000 -> ~7.1      n_llm = 40,000 -> ~8.4
Scale n_llm (Phase 1) to audit higher epsilon; auditing eps=16 would need n_llm ~ e^16.

Multiple J (--num_exemplars 1 2 3 4) are audited from one pass of LLM votes; each J
uses gamma_conf/len(J) (Bonferroni), so the reported best over J is a valid bound.

Usage:
    python ksa_active.py --data_path /path/to/samsum/train.csv --dataset_name samsum \
        --n_llm 40000 --n_sample 800000 --epsilons 1 2 4 8 --num_exemplars 1 2 3 4
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
    parse_yes_no, pure_dp_eps_lower_bound,
)
import auditing_utils


# ==============================================================================
# KSA Core
# ==============================================================================
_greedy_votes = {}   # prompt -> vote; reused across calls under greedy decoding


def collect_votes(texts, query_text, batch_size, desc=None):
    """Yes(1)/No(0) vote of a single-exemplar voter for each text, in input order.
    Under greedy decoding identical prompts give identical votes, so each distinct
    prompt is sent to the LLM only once (and cached for later calls).
    batch_size <= 1 disables both batching and the cache (one LLM call per vote)."""
    prompts = [build_membership_prompt(t, query_text) for t in texts]
    sampling = auditing_utils.generation_temperature and auditing_utils.generation_temperature > 0
    if sampling or batch_size <= 1:
        responses = generate_responses(prompts, batch_size, desc)
        return np.array([parse_yes_no(r) == "Yes" for r in responses], dtype=float)
    missing = [p for p in dict.fromkeys(prompts) if p not in _greedy_votes]
    for p, r in zip(missing, generate_responses(missing, batch_size, desc)):
        _greedy_votes[p] = parse_yes_no(r) == "Yes"
    return np.array([_greedy_votes[p] for p in prompts], dtype=float)


# ==============================================================================
# Phase 1: Collect clean vote histograms for every J from one pass of votes
# ==============================================================================
def collect_snapshots(all_texts, canary_text, canary_index, js, n_llm, rng, f, batch_size,
                      filter_yes_voters=False, fixed_context=False):
    """Each KSA voter sees only its own exemplar, so a J-voter histogram is the sum
    of J independent single-exemplar votes. Per snapshot i we draw j_max distinct
    non-canary exemplars W_i and one canary vote c_i, then for each J:
        without_J = votes(W_i[:J])
        with_J    = c_i + votes(W_i[:J-1])   (W_i[J-1] swapped for the canary)
    The audits for different J share these votes; the Bonferroni split of
    gamma_conf across J does not need them to be independent.

    filter_yes_voters: the auditor picks the dataset, so drop every non-canary text
    whose (pilot) vote is Yes before drawing exemplars. This removes the FPR floor
    that false-Yes voters put on the "without" world. Under greedy decoding the
    pilot votes are exact and cached (no extra LLM cost); under sampling the pilot
    is one draw per text and the audit votes below are fresh samples.

    fixed_context: draw W once and reuse it for every snapshot of the run, so the
    audit targets one fixed pair of neighbouring contexts."""
    remaining = [i for i in range(len(all_texts)) if i != canary_index]
    j_max = max(js)

    if filter_yes_voters:
        pilot = collect_votes([all_texts[i] for i in remaining], canary_text,
                              batch_size, desc="Pilot votes (filter Yes voters)")
        n_before = len(remaining)
        remaining = [i for i, v in zip(remaining, pilot) if v == 0]
        print(f"[Filter] Dropped {n_before - len(remaining)}/{n_before} non-canary texts "
              f"that voted Yes; pool size now {len(remaining)}", file=f)
        if len(remaining) < j_max:
            raise ValueError(f"Only {len(remaining)} No-voting texts left, need J={j_max}")

    if fixed_context:
        draws = [rng.sample(remaining, j_max)] * n_llm
    else:
        draws = [rng.sample(remaining, j_max) for _ in range(n_llm)]
    other_votes = collect_votes([all_texts[i] for d in draws for i in d], canary_text,
                                batch_size, desc="Votes (non-canary exemplars)").reshape(n_llm, j_max)
    canary_votes = collect_votes([canary_text] * n_llm, canary_text,
                                 batch_size, desc="Votes (canary exemplar)")
    print(f"Yes-rate: canary voter={canary_votes.mean():.6f}, "
          f"non-canary voters={other_votes.mean():.6f}", file=f)

    cum = np.cumsum(other_votes, axis=1)                             # (n_llm, j_max)
    hists = {}
    for J in js:
        yes_without = cum[:, J - 1]
        yes_with = canary_votes + (cum[:, J - 2] if J > 1 else 0.0)
        h_with = np.stack([yes_with, J - yes_with], axis=1)
        h_without = np.stack([yes_without, J - yes_without], axis=1)
        print(f"[J={J}] Mean histogram (with): {h_with.mean(axis=0)}, "
              f"(without): {h_without.mean(axis=0)}", file=f)
        print(f"[J={J}] Max per-dim variance (with): {h_with.var(axis=0).max():.6f}, "
              f"(without): {h_without.var(axis=0).max():.6f}", file=f)
        hists[J] = (h_with, h_without)
    return hists


# ==============================================================================
# Phase 2: m fresh Gumbel-max draws PER snapshot (noise averaging, not bootstrap)
# ==============================================================================
def simulate_per_snapshot(clean_histograms, epsilon, m):
    """Draw m fresh Gumbel-max samples per snapshot.

    Returns:
      ghat:   (n_llm,)   per-snapshot canary-selection rate in [0,1] (black-box stat)
      scores: (n_llm, m) per-snapshot advantage scores (yes - no) for white-box
    """
    sensitivity = 1.0
    hists = np.asarray(clean_histograms, dtype=float)                 # (n_llm, C)
    base = hists * epsilon / (2.0 * sensitivity)                      # (n_llm, C)
    noise = np.random.gumbel(0.0, 1.0, (hists.shape[0], m, hists.shape[1]))
    noisy = base[:, None, :] + noise                                 # (n_llm, m, C)
    selections = np.argmax(noisy, axis=2)                            # (n_llm, m)
    ghat = np.mean(selections == 0, axis=1)                          # (n_llm,)
    scores = noisy[:, :, 0] - noisy[:, :, 1]                         # (n_llm, m)
    return ghat, scores


# ==============================================================================
# Phase 3: Empirical-Bernstein audit on the n_llm per-snapshot rates
# ==============================================================================
def sweep_rate_stats(scores, thresholds):
    """Mean and sample variance over snapshots of the rate P(score > t), for every t.
    One sort plus cumulative sums: adding the (c+1)-th exceedance of a snapshot
    raises its squared count by 2c+1, so no per-threshold pass over the scores."""
    n, m = scores.shape
    ranked = -np.sort(-scores, axis=1).ravel()
    increments = np.tile(2.0 * np.arange(m) + 1.0, n)
    order = np.argsort(ranked)                                       # ascending scores
    ascending = ranked[order]
    cum_sq = np.concatenate(([0.0], np.cumsum(increments[order][::-1])))
    k = scores.size - np.searchsorted(ascending, thresholds, side="right")
    mean = k / (n * m)
    var = (cum_sq[k] / m ** 2 - n * mean ** 2) / (n - 1)
    return mean, np.maximum(var, 0.0)


def full_threshold_sweep(scores_with, scores_without, n_llm, delta, log_g, range_term):
    """White-box sweep over every observed score. Returns (best eps, best threshold)."""
    thresholds = np.unique(np.concatenate((scores_with.ravel(), scores_without.ravel(), [0.0])))
    mean_w, var_w = sweep_rate_stats(scores_with, thresholds)
    mean_wo, var_wo = sweep_rate_stats(scores_without, thresholds)
    tpr_low = np.maximum(0.0, mean_w - (np.sqrt(2.0 * var_w * log_g / n_llm) + range_term))
    fpr_up = np.minimum(1.0, mean_wo + np.sqrt(2.0 * var_wo * log_g / n_llm) + range_term)
    fnr_up = 1.0 - tpr_low
    valid = (fpr_up < 1) & (fnr_up < 1) & (fpr_up > 0) & (fnr_up > 0)
    if not valid.any():
        return -float("inf"), None
    fpr, fnr, kept = fpr_up[valid], fnr_up[valid], thresholds[valid]
    num1, num2 = 1.0 - fnr - delta, 1.0 - fpr - delta
    with np.errstate(divide="ignore", invalid="ignore"):
        eps1 = np.where(num1 > 0, np.log(num1 / fpr), 0.0)
        eps2 = np.where(num2 > 0, np.log(num2 / fnr), 0.0)
    eps = np.maximum(np.maximum(eps1, eps2), 0.0)
    best = int(np.argmax(eps))
    return float(eps[best]), float(kept[best])


def run_audit(ghat_with, scores_with, ghat_without, scores_without,
              n_llm, m, delta, gamma_conf, f, n_grid=400):
    log_g = math.log(2.0 / gamma_conf)
    range_term = 7.0 * log_g / (3.0 * (n_llm - 1))

    def eb_delta(sigma2):
        # Maurer-Pontil empirical-Bernstein deviation for iid [0,1] variables.
        return math.sqrt(2.0 * sigma2 * log_g / n_llm) + range_term

    ceiling = math.log((1.0 - range_term) / range_term) if 0.0 < range_term < 1.0 else float("inf")
    print(f"\n--- Empirical-Bernstein on n_llm={n_llm} (m={m} noise/snapshot, gamma={gamma_conf}) ---", file=f)
    print(f"  Range term (m-independent, sets the tail floor): {range_term:.6f}", file=f)
    print(f"  Deep-tail eps ceiling (n_llm-limited): {ceiling:.4f}", file=f)

    # === Black-box (discrete EM output: Yes=0) ===
    tpr_hat = float(ghat_with.mean())
    fpr_hat = float(ghat_without.mean())
    s2_with = float(ghat_with.var(ddof=1))
    s2_without = float(ghat_without.var(ddof=1))
    tpr_low = max(0.0, tpr_hat - eb_delta(s2_with))
    fpr_up = min(1.0, fpr_hat + eb_delta(s2_without))
    eps_bb = pure_dp_eps_lower_bound(fpr_up, 1.0 - tpr_low, delta)
    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f} (var={s2_with:.6f}), "
          f"FPR_hat={fpr_hat:.6f} (var={s2_without:.6f})", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.6f}", file=f)
    print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds on the advantage score) ===
    # For a fixed threshold t: per-snapshot rate ghat_{t,i} = P_Z(score_i > t),
    # estimated from the m draws; bound E_X[ghat_t] by empirical Bernstein.
    # A threshold grid keeps this O(n_grid * n_llm * m); any fixed t gives a valid
    # lower bound, so the max over the grid is a valid (slightly conservative) bound.
    # Per-threshold confidence (no multiple-testing correction), consistent with the
    # other auditors in this repo.
    if n_grid <= 0:
        max_eps, best_threshold = full_threshold_sweep(
            scores_with, scores_without, n_llm, delta, log_g, range_term)
    else:
        pooled = np.concatenate((scores_with.ravel(), scores_without.ravel()))
        qs = np.quantile(pooled, np.linspace(0.0, 1.0, n_grid))
        grid = np.unique(np.concatenate((qs, [0.0])))

        max_eps = -float("inf")
        best_threshold = None
        for t in grid:
            gw = (scores_with > t).mean(axis=1)                          # (n_llm,)
            gwo = (scores_without > t).mean(axis=1)
            tpr_low_t = max(0.0, float(gw.mean()) - eb_delta(float(gw.var(ddof=1))))
            fpr_up_t = min(1.0, float(gwo.mean()) + eb_delta(float(gwo.var(ddof=1))))
            fnr_up_t = 1.0 - tpr_low_t
            if fpr_up_t >= 1 or fnr_up_t >= 1 or fpr_up_t <= 0 or fnr_up_t <= 0:
                continue
            cur_eps = pure_dp_eps_lower_bound(fpr_up_t, fnr_up_t, delta)
            if cur_eps > max_eps:
                max_eps, best_threshold = cur_eps, float(t)

    eps_wb = max(max_eps, 0.0) if best_threshold is not None else 0.0
    if best_threshold is not None:
        print(f"\n[White-box] Optimal threshold={best_threshold:.4f}", file=f)
        print(f"[White-box] eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="KSA Active Auditing (per-snapshot noise averaging + empirical Bernstein)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_exemplars", type=int, nargs="+", default=[1],
                        help="One or more J values. Each J is audited at gamma_conf/len(J) "
                             "(Bonferroni), so the best over J is a valid (1-gamma_conf) bound.")
    parser.add_argument("--n_llm", type=int, default=20000,
                        help="Number of clean snapshots (Phase 1). Also caps the "
                             "auditable epsilon at ~log(n_llm).")
    parser.add_argument("--n_sample", type=int, default=400000,
                        help="Total noise budget (Phase 2). Split as m = n_sample // n_llm "
                             "fresh Gumbel draws per snapshot; tightens the bulk, not the tail.")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--n_grid", type=int, default=400, help="White-box threshold grid size (0 = sweep every observed score)")
    parser.add_argument("--fixed_context", action="store_true",
                        help="Draw the context once per run instead of once per snapshot")
    parser.add_argument("--output_dir", type=str, default="results_ksa_active")
    parser.add_argument("--output_prefix", type=str, default="ksa-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--canary_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, default="semantic_similarity")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    parser.add_argument("--filter_yes_voters", action="store_true",
                        help="Draw non-canary exemplars only from texts whose pilot vote is No "
                             "(auditor's choice of dataset; removes the false-Yes FPR floor)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM generate call")
    args = parser.parse_args()

    auditing_utils.attack_mode = args.attack_mode
    auditing_utils.max_new_tokens_global = args.max_new_tokens
    auditing_utils.generation_temperature = args.temperature
    load_hf_model(args.model_path)
    all_texts = load_texts(args.data_path, args.dataset_name)
    if not all_texts:
        print("Error: no texts loaded.")
        return

    m = max(1, args.n_sample // args.n_llm)
    js = sorted(set(args.num_exemplars))
    gamma_per_j = args.gamma_conf / len(js)
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
            print("KSA Active Auditing (per-snapshot noise averaging + empirical Bernstein)", file=f)
            print(f"Mechanism: Exponential (Gumbel-max), pure DP", file=f)
            print(f"Estimator: Pure DP", file=f)
            print(f"Error: Empirical Bernstein on n_llm per-snapshot rates (m noise/snapshot)", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, m={m}, J={js}", file=f)
            print(f"filter_yes_voters={args.filter_yes_voters}", file=f)
            print(f"fixed_context={args.fixed_context}", file=f)
            print(f"gamma_conf={args.gamma_conf} split over {len(js)} J (Bonferroni): "
                  f"{gamma_per_j:.4g} per J", file=f)
            print("=" * 60, file=f)

            t1_start = time.time()
            hists = collect_snapshots(
                all_texts, canary, idx, js, args.n_llm, rng, f, args.batch_size,
                filter_yes_voters=args.filter_yes_voters, fixed_context=args.fixed_context
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Phase 1: {t1_time:.1f}s", file=f)

            for epsilon in args.epsilons:
                best_bb, best_wb = (-float("inf"), None), (-float("inf"), None)
                for J in js:
                    print(f"\n{'='*40} eps_theory={epsilon}, J={J} {'='*40}", file=f)
                    hists_with, hists_without = hists[J]

                    t2_start = time.time()
                    ghat_with, scores_with = simulate_per_snapshot(hists_with, epsilon, m)
                    ghat_without, scores_without = simulate_per_snapshot(hists_without, epsilon, m)
                    t2_time = time.time() - t2_start

                    t3_start = time.time()
                    eps_bb, eps_wb = run_audit(
                        ghat_with, scores_with, ghat_without, scores_without,
                        n_llm=args.n_llm, m=m, delta=args.delta,
                        gamma_conf=gamma_per_j, f=f, n_grid=args.n_grid
                    )
                    t3_time = time.time() - t3_start

                    print(f"\n[TIMING] Phase 2: {t2_time:.1f}s, Phase 3: {t3_time:.1f}s", file=f)
                    print(f"[RESULT] eps_theory={epsilon}, J={J}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                    best_bb = max(best_bb, (eps_bb, J))
                    best_wb = max(best_wb, (eps_wb, J))

                print(f"\n[BEST] eps_theory={epsilon}: bb={best_bb[0]:.4f} (J={best_bb[1]}), "
                      f"wb={best_wb[0]:.4f} (J={best_wb[1]}) "
                      f"[Bonferroni over {len(js)} J, gamma/J={gamma_per_j:.4g}]", file=f)
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
