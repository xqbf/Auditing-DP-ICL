"""
PoE Active Auditing (honest two-phase: per-snapshot noise averaging + empirical Bernstein).

Phase 1 (expensive, LLM): collect n_llm clean utility vectors, each over an
    independently sampled exemplar context, so the audit reflects the genuine
    randomness of the DP-ICL input distribution (NOT a fixed / cherry-picked context).
Phase 2 (cheap, noise): for each epsilon, draw m = n_sample / n_llm fresh
    Gumbel-max samples PER snapshot (scale 2*gamma_clip/epsilon) and average them
    into a per-snapshot canary-selection rate ghat_i in [0,1] (and per-snapshot
    advantage scores for the white-box sweep). This is the correct use of the noise
    budget -- it drives down the within-snapshot noise variance E_X[g(1-g)], NOT a
    bootstrap resample that would over-count n_sample as the effective sample size.
Phase 3 (estimate): bound TPR/FPR with empirical Bernstein (Maurer-Pontil) on the
    n_llm iid per-snapshot rates ghat_i, using their REAL empirical variance, then
    convert to epsilon via the pure-DP estimator (PoE = exponential mechanism).

Fundamental limit (pure-DP / exponential mechanism): the recoverable epsilon is
floored by n_llm -- certifying a false-positive rate ~e^{-eps} needs ~e^{eps}
snapshots. Deep-tail ceiling ~= log((1-r)/r), r = 7*log(2/gamma)/(3*(n_llm-1)):
  n_llm = 2,000 -> ~5.4     n_llm = 20,000 -> ~7.8     n_llm = 40,000 -> ~8.4
Scale n_llm (Phase 1) to audit higher epsilon.

Multiple J (--num_exemplars 1 2 3 4) are audited from one batched pass of expert
log-probs; each J uses gamma_conf/len(J) (Bonferroni), so the best over J is valid.

Usage:
    python poe_active.py --data_path /path/to/data.csv --n_llm 20000 --epsilons 1 2 4 8 \
        --num_exemplars 1 2 3 4
"""

import numpy as np
import math
import argparse
import os
import random
import time
import torch

from auditing_utils import (
    load_hf_model, load_texts_from_csv, build_poe_expert_prompt,
    get_label_token_ids, get_log_probs_for_labels_batch, pure_dp_eps_lower_bound,
)
import auditing_utils


# ==============================================================================
# PoE Mechanism
# ==============================================================================
def clip_log_probs(log_probs: np.ndarray, gamma: float, mode: str = "abstain") -> np.ndarray:
    """Clip log-probabilities to [-gamma, 0].
    abstain: values outside -> 0 (original PoE, same as poe_passive.py).
    clamp:   values outside -> nearest bound."""
    if mode == "clamp":
        return np.clip(log_probs, -gamma, 0.0)
    result = log_probs.copy()
    result[(log_probs < -gamma) | (log_probs > 0.0)] = 0.0
    return result


class PoEClassifier:
    def __init__(self, delta, gamma=6.0, clip="abstain"):
        self.class_labels = ["Yes", "No"]
        self.num_classes = 2
        self.label_to_id = {"Yes": 0, "No": 1}
        self.delta = delta
        self.gamma = gamma          # log-prob clipping bound (sensitivity parameter)
        self.clip = clip            # out-of-range handling: "abstain" or "clamp"
        self.label_token_ids = None

    def init_label_tokens(self):
        self.label_token_ids = get_label_token_ids(self.class_labels)

    def expert_utilities(self, texts, query_text, batch_size, desc=None):
        """Clipped log-prob vector of a single-exemplar expert for each text, (len, C)."""
        prompts = [build_poe_expert_prompt(t, query_text) for t in texts]
        log_probs = get_log_probs_for_labels_batch(prompts, self.label_token_ids, batch_size, desc)
        return clip_log_probs(log_probs.astype(float), self.gamma, self.clip)


# ==============================================================================
# Phase 1: Collect clean utility vectors for every J from one pass of experts
# ==============================================================================
def collect_snapshots(poe, canary_text, pool_texts, canary_index, js, n_llm, rng, f,
                      batch_size, fixed_context=False):
    """Each PoE expert sees only its own exemplar, so a J-expert utility vector is
    the sum of J independent single-exemplar vectors. Per snapshot i we draw j_max
    distinct non-canary exemplars W_i, then for each J:
        without_J = sum experts(W_i[:J])
        with_J    = expert(canary) + sum experts(W_i[:J-1])   (W_i[J-1] swapped out)
    The expert forward pass is deterministic, so every distinct exemplar is run
    through the LLM once (batched) and reused across snapshots and J. The audits
    for different J share these vectors; the Bonferroni split of gamma_conf across
    J does not need them to be independent.

    fixed_context: draw W once and reuse it for every snapshot of the run, so the
    audit targets one fixed pair of neighbouring contexts."""
    remaining = [i for i in range(len(pool_texts)) if i != canary_index]
    j_max = max(js)

    if fixed_context:
        draws = np.array([rng.sample(remaining, j_max)] * n_llm)       # (n_llm, j_max)
    else:
        draws = np.array([rng.sample(remaining, j_max) for _ in range(n_llm)])  # (n_llm, j_max)
    other = poe.expert_utilities([pool_texts[i] for i in draws.ravel()], canary_text, batch_size,
                                 desc="Experts (non-canary exemplars)")
    other = other.reshape(n_llm, j_max, -1)                            # (n_llm, j_max, C)
    canary_utils = poe.expert_utilities([canary_text] * n_llm, canary_text, batch_size,
                                        desc="Experts (canary exemplar)")  # (n_llm, C)
    print(f"Canary expert utility (mean): {canary_utils.mean(axis=0)}", file=f)
    print(f"Distinct non-canary exemplars used: {len(np.unique(draws))}", file=f)

    cum = np.cumsum(other, axis=1)                                     # (n_llm, j_max, C)
    vecs = {}
    for J in js:
        v_without = cum[:, J - 1]
        v_with = canary_utils + (cum[:, J - 2] if J > 1 else 0.0)
        assert np.isfinite(v_with).all() and np.isfinite(v_without).all(), \
            "Non-finite utility vectors (check logit temperature)"
        print(f"[J={J}] Mean utility (with): {v_with.mean(axis=0)}, "
              f"(without): {v_without.mean(axis=0)}", file=f)
        print(f"[J={J}] Max per-dim variance (with): {v_with.var(axis=0).max():.6e}, "
              f"(without): {v_without.var(axis=0).max():.6e}", file=f)
        vecs[J] = (v_with, v_without)
    return vecs


# ==============================================================================
# Phase 2: m fresh Gumbel-max draws PER snapshot (noise averaging, not bootstrap)
# ==============================================================================
def simulate_per_snapshot(clean_vectors, epsilon, gamma_clip, m, canary_class_id=0):
    """Draw m fresh Gumbel-max samples per snapshot.

    Returns:
      ghat:   (n_llm,)   per-snapshot canary-selection rate in [0,1] (black-box stat)
      scores: (n_llm, m) per-snapshot advantage scores util[canary] - max(others)
    """
    gumbel_scale = 2.0 * gamma_clip / epsilon
    vecs = np.asarray(clean_vectors, dtype=float)                     # (n_llm, C)
    noise = np.random.gumbel(0.0, gumbel_scale, (vecs.shape[0], m, vecs.shape[1]))
    noisy = vecs[:, None, :] + noise                                 # (n_llm, m, C)
    selections = np.argmax(noisy, axis=2)                            # (n_llm, m)
    ghat = np.mean(selections == canary_class_id, axis=1)            # (n_llm,)
    canary = noisy[:, :, canary_class_id]                            # (n_llm, m)
    others = np.max(np.delete(noisy, canary_class_id, axis=2), axis=2)
    scores = canary - others                                        # (n_llm, m)
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

    # === Black-box (threshold at 0: argmax == canary) ===
    tpr_hat = float(ghat_with.mean())
    fpr_hat = float(ghat_without.mean())
    s2_with = float(ghat_with.var(ddof=1))
    s2_without = float(ghat_without.var(ddof=1))
    tpr_low = max(0.0, tpr_hat - eb_delta(s2_with))
    fpr_up = min(1.0, fpr_hat + eb_delta(s2_without))
    eps_bb = pure_dp_eps_lower_bound(fpr_up, 1.0 - tpr_low, delta)
    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f} (var={s2_with:.6e}), "
          f"FPR_hat={fpr_hat:.6f} (var={s2_without:.6e})", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.6f}", file=f)
    print(f"[Black-box] eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds on the advantage score) ===
    # For a fixed threshold t: per-snapshot rate ghat_{t,i} = P_Z(score_i > t),
    # estimated from the m draws; bound E_X[ghat_t] by empirical Bernstein. A
    # threshold grid keeps this O(n_grid * n_llm * m); any fixed t gives a valid
    # lower bound, so the max over the grid is valid (slightly conservative).
    # Per-threshold confidence (no multiple-testing correction), consistent with
    # the other auditors in this repo.
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
        description="PoE Active Auditing (per-snapshot noise averaging + empirical Bernstein)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--text_col", type=int, default=1)
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--num_exemplars", type=int, nargs="+", default=[1],
                        help="One or more J values (number of experts). Each J is audited at "
                             "gamma_conf/len(J) (Bonferroni), so the best over J is a valid "
                             "(1-gamma_conf) bound.")
    parser.add_argument("--n_llm", type=int, default=50000,
                        help="Number of clean snapshots (Phase 1). Also caps the "
                             "auditable epsilon at ~log(n_llm).")
    parser.add_argument("--n_sample", type=int, default=400000,
                        help="Total noise budget (Phase 2). Split as m = n_sample // n_llm "
                             "fresh Gumbel draws per snapshot; tightens the bulk, not the tail.")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_clip", type=float, default=4.5, help="Log-prob clipping bound")
    parser.add_argument("--clip", choices=["abstain", "clamp"], default="abstain",
                        help="Out-of-range log-probs: abstain (-> 0, original PoE) or clamp (-> bound)")
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--n_grid", type=int, default=400, help="White-box threshold grid size (0 = sweep every observed score)")
    parser.add_argument("--fixed_context", action="store_true",
                        help="Draw the context once per run instead of once per snapshot")
    parser.add_argument("--output_dir", type=str, default="results_poe_active")
    parser.add_argument("--output_prefix", type=str, default="poe-active")
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
        canary_text = all_texts[idx]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        t_run = time.time()
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("PoE Active Auditing (per-snapshot noise averaging + empirical Bernstein)", file=f)
            print(f"Canary: '{canary_text[:100]}...'", file=f)
            print(f"Mechanism: Exponential (Gumbel-max), pure DP", file=f)
            print(f"Clip: {args.clip} to [-{args.gamma_clip}, 0], logit temperature={args.temperature}", file=f)
            print(f"Estimator: Pure DP", file=f)
            print(f"Error: Empirical Bernstein on n_llm per-snapshot rates (m noise/snapshot)", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, m={m}, J={js}", file=f)
            print(f"fixed_context={args.fixed_context}", file=f)
            print(f"gamma_conf={args.gamma_conf} split over {len(js)} J (Bonferroni): "
                  f"{gamma_per_j:.4g} per J", file=f)
            print("=" * 60, file=f)

            poe = PoEClassifier(args.delta, args.gamma_clip, args.clip)
            poe.init_label_tokens()

            t1_start = time.time()
            vecs = collect_snapshots(
                poe, canary_text, all_texts, idx, js, args.n_llm, rng, f, args.batch_size,
                fixed_context=args.fixed_context
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Phase 1: {t1_time:.1f}s", file=f)

            for epsilon in args.epsilons:
                best_bb, best_wb = (-float("inf"), None), (-float("inf"), None)
                for J in js:
                    print(f"\n{'='*40} eps_theory={epsilon}, J={J} {'='*40}", file=f)
                    vecs_with, vecs_without = vecs[J]

                    t2_start = time.time()
                    ghat_with, scores_with = simulate_per_snapshot(
                        vecs_with, epsilon, args.gamma_clip, m)
                    ghat_without, scores_without = simulate_per_snapshot(
                        vecs_without, epsilon, args.gamma_clip, m)
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
