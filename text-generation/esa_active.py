"""
ESA Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean mean embeddings via real LLM calls.
Stage 2: Bootstrap n_sample noisy embeddings by resampling + Gaussian noise.
Stage 3: Compute TPR/FPR with Bernstein (snapshot) + Clopper-Pearson (MC),
         then convert to epsilon via GDP estimator.

Sensitivity = 2/K (FIXED). Analytic Gaussian (Balle & Wang 2018).

Usage:
    python esa_active.py --data_path /path/to/samsum/train.csv
"""

import numpy as np
import math
import argparse
import os
import random
import time
import csv
import sys
import torch
from tqdm import tqdm
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from auditing_utils import (
    clopper_pearson, gdp_to_eps, analytic_gauss_sigma,
    build_esa_generator_prompt, load_data_csv, DATASET_CONFIGS,
)

# ==============================================================================
# Global model state
# ==============================================================================
tokenizer = None
model = None
# LLM generation temperature. 0.0 -> greedy decoding (do_sample=False);
# any value > 0 enables sampling at that temperature.
generation_temperature = 0.0


def load_hf_model(model_path):
    global tokenizer, model
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


# ==============================================================================
# ESA helpers
# ==============================================================================
def generate_signal_sentence(contexts, canary, y1_target, y2_control,
                             max_new_tokens=400):
    """Generate a constrained signal sentence from LLM (temperature 0)."""
    exemplar_context = "\n".join(contexts) if contexts else ""
    prompt = build_esa_generator_prompt(canary, y1_target, y2_control, exemplar_context)
    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            **({"do_sample": True, "temperature": generation_temperature}
               if generation_temperature and generation_temperature > 0
               else {"do_sample": False}),
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    if y1_target in response:
        return y1_target
    if y2_control in response:
        return y2_control
    return response


_greedy_outputs = {}   # prompt -> mapped output; reused across calls under greedy decoding


def generate_signal_sentences(context_texts, canary, y1_target, y2_control, batch_size,
                              desc=None, max_new_tokens=400):
    """Batched generate_signal_sentence (one single-exemplar context each), in input
    order. Under greedy decoding identical prompts give identical outputs, so each
    distinct prompt is sent to the LLM only once (and cached for later calls).
    Under sampling (temperature > 0) nothing is deduplicated or cached: every prompt
    gets its own independent sample, exactly as in the one-at-a-time path.
    batch_size <= 1: plain one-at-a-time calls, no cache (original behaviour)."""
    if batch_size <= 1:
        return [generate_signal_sentence([c], canary, y1_target, y2_control, max_new_tokens)
                for c in tqdm(context_texts, desc=desc, disable=desc is None)]
    raws = [tokenizer.apply_chat_template(
                [{"role": "user", "content": build_esa_generator_prompt(canary, y1_target, y2_control, c)}],
                tokenize=False, add_generation_prompt=True)
            for c in context_texts]
    greedy = not (generation_temperature and generation_temperature > 0)
    todo = [r for r in dict.fromkeys(raws) if r not in _greedy_outputs] if greedy else raws
    gen_kwargs = ({"do_sample": False} if greedy
                  else {"do_sample": True, "temperature": generation_temperature})
    order = sorted(range(len(todo)), key=lambda i: len(todo[i]))
    tokenizer.padding_side = "left"   # decoder-only: new tokens must follow the prompt
    outputs_todo = [None] * len(todo)
    for start in tqdm(range(0, len(order), batch_size), desc=desc, disable=desc is None):
        idx = order[start:start + batch_size]
        inputs = tokenizer([todo[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
        texts = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:],
                                       skip_special_tokens=True)
        for i, t in zip(idx, texts):
            t = t.strip()
            outputs_todo[i] = y1_target if y1_target in t else y2_control if y2_control in t else t
    if not greedy:
        return outputs_todo
    _greedy_outputs.update(zip(todo, outputs_todo))
    return [_greedy_outputs[r] for r in raws]


def subset_contexts(exemplars, num_subsets):
    """The single context each subset generates from (first exemplar of its
    partition; None for an empty one). Same np.random draw as the original loop."""
    shuffled = np.random.permutation(len(exemplars))
    partitions = np.array_split(shuffled, num_subsets)
    return [exemplars[p[0]] if len(p) > 0 else None for p in partitions]


def clean_mean_embeddings(embedding_model, contexts_per_snapshot, num_subsets, canary,
                          y1_target, y2_control, batch_size, desc=None):
    """Mean of normalized subset-output embeddings per snapshot, shape (n, dim).
    All subset prompts of all snapshots go through the LLM in batches."""
    flat = [(s, c) for s, ctxs in enumerate(contexts_per_snapshot) for c in ctxs if c is not None]
    outputs = generate_signal_sentences([c for _, c in flat], canary, y1_target, y2_control,
                                        batch_size, desc=desc)
    if batch_size <= 1:
        embs = np.array([embedding_model.encode(o, convert_to_numpy=True) for o in outputs])
    else:
        embs = embedding_model.encode(outputs, convert_to_numpy=True, batch_size=batch_size)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = np.where(norms > 0, embs / np.where(norms > 0, norms, 1.0), 0.0)
    sums = np.zeros((len(contexts_per_snapshot), embedding_model.get_sentence_embedding_dimension()))
    np.add.at(sums, [s for s, _ in flat], embs)
    return sums / num_subsets


# ==============================================================================
# Stage 1: Collect clean mean embeddings
# ==============================================================================
def collect_snapshots(embedding_model, all_data, canary_index, canary_exemplar,
                      canary_fact, num_exemplars, num_subsets,
                      y1_target, y2_control, n_llm, dataset_type, rng, f,
                      replicate=1, batch_size=64):
    remaining = [i for i in range(len(all_data)) if i != canary_index]

    def sample_exemplars_with():
        indices = rng.sample(remaining, num_exemplars - 1)
        if dataset_type == "xsum":
            base = [all_data[i][2] for i in indices]
        else:
            base = [all_data[i][1] + " " + all_data[i][2] for i in indices]
        pos = random.randint(0, len(base))
        base.insert(pos, canary_exemplar)
        return base

    def sample_exemplars_without():
        indices = rng.sample(remaining, num_exemplars)
        if dataset_type == "xsum":
            return [all_data[i][2] for i in indices]
        return [all_data[i][1] + " " + all_data[i][2] for i in indices]

    n_unique = n_llm // replicate

    # Draw every snapshot's exemplars and subset partition in the original order
    # (rng/random for exemplars, np.random for partitions), then run the LLM batched.
    ctx_with = [subset_contexts(sample_exemplars_with(), num_subsets) for _ in range(n_unique)]
    ctx_without = [subset_contexts(sample_exemplars_without(), num_subsets) for _ in range(n_unique)]

    uniq_with = clean_mean_embeddings(embedding_model, ctx_with, num_subsets, canary_fact,
                                      y1_target, y2_control, batch_size,
                                      desc="Snapshots (with canary)")
    uniq_without = clean_mean_embeddings(embedding_model, ctx_without, num_subsets, canary_fact,
                                         y1_target, y2_control, batch_size,
                                         desc="Snapshots (without canary)")
    vectors_with = [vec for vec in uniq_with for _ in range(replicate)]
    vectors_without = [vec for vec in uniq_without for _ in range(replicate)]

    print(f"Mean embedding L2 (with vs without): "
          f"{np.linalg.norm(np.mean(vectors_with, axis=0) - np.mean(vectors_without, axis=0)):.6f}", file=f)
    print(f"Unique snapshots: {n_unique}, replicate: {replicate}, total: {len(vectors_with)}", file=f)
    return vectors_with, vectors_without


# ==============================================================================
# Stage 2: Bootstrap with Gaussian noise (score-space, memory-efficient)
# ==============================================================================
def bootstrap_simulate_gaussian_scores(collected_vectors, sigma, n_sample,
                                       direction):
    """Resample + Gaussian noise, but only return projected 1-D scores.

    direction: unit-norm vector (y1 - y2 direction).
    Score = dot(resampled + noise, direction)
          = dot(resampled, direction) + N(0, sigma * ||direction||)
    Since direction is passed pre-normalized, noise_std = sigma.
    """
    vecs = np.array(collected_vectors)
    base_scores = vecs @ direction                          # (n_vec,)
    idx = np.random.choice(len(base_scores), n_sample, replace=True)
    scores = base_scores[idx]                               # (n_sample,)
    scores += np.random.normal(0, sigma, n_sample)          # scalar noise
    return scores


# ==============================================================================
# Stage 3: Audit with two-layer error bound
# ==============================================================================
def run_active_audit(scores_with, scores_without,
                     n_llm, n_sample, delta, gamma_conf, f,
                     snapshot_gamma_frac=0.5):
    T_audit = len(scores_with)

    # --- Bernstein snapshot correction + Clopper-Pearson for MC ---
    sigma2_with = 0.0
    sigma2_without = 0.0

    gamma_snap = gamma_conf * snapshot_gamma_frac
    gamma_mc = gamma_conf * (1.0 - snapshot_gamma_frac)
    log_snap = math.log(2.0 / gamma_snap)

    bern_with = (math.sqrt(2.0 * sigma2_with * log_snap / n_llm)
                 + 7.0 * log_snap / (3.0 * (n_llm - 1)))
    bern_without = (math.sqrt(2.0 * sigma2_without * log_snap / n_llm)
                    + 7.0 * log_snap / (3.0 * (n_llm - 1)))

    cp_confidence = 1.0 - gamma_mc

    print(f"\n--- Error Bounds (gamma={gamma_conf}, snapshot_gamma_frac={snapshot_gamma_frac}) ---", file=f)
    print(f"  Bernstein_1 (snapshot): {bern_with:.6f}", file=f)
    print(f"  Bernstein_0 (snapshot): {bern_without:.6f}", file=f)
    print(f"  Clopper-Pearson confidence: {cp_confidence}", file=f)

    def get_tpr_low_fpr_up(tp_count, fp_count, n):
        cp_tpr_lo, _ = clopper_pearson(tp_count, n, cp_confidence)
        _, cp_fpr_hi = clopper_pearson(fp_count, n, cp_confidence)
        tpr_low = max(0.0, cp_tpr_lo - bern_with)
        fpr_up = min(1.0, cp_fpr_hi + bern_without)
        return tpr_low, fpr_up

    # === Black-box (mapping attack: sim_y1 > sim_y2) ===
    tp_count = int(np.sum(scores_with > 0))
    fp_count = int(np.sum(scores_without > 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low, fpr_up = get_tpr_low_fpr_up(tp_count, fp_count, T_audit)

    mu_bb = 0.0
    if 0 < fpr_up < 1 and 0 < (1 - tpr_low) < 1:
        mu_bb = max(norm.ppf(tpr_low) - norm.ppf(fpr_up), 0)
    eps_bb = gdp_to_eps(mu_bb, delta)

    print(f"\n[Black-box] TPR_hat={tpr_hat:.4f}, FPR_hat={fpr_hat:.4f}", file=f)
    print(f"[Black-box] TPR_low={tpr_low:.4f}, FPR_up={fpr_up:.4f}", file=f)
    print(f"[Black-box] mu_lower={mu_bb:.4f}, eps_lower={eps_bb:.4f}", file=f)

    # === White-box (sweep thresholds) ===
    obs_with = np.sort(scores_with)
    obs_without = np.sort(scores_without)
    candidates = np.unique(np.concatenate((obs_with, obs_without)))

    max_mu = -float("inf")
    best_threshold = None

    for threshold in candidates:
        fp = T_audit - int(np.searchsorted(obs_without, threshold, side="right"))
        fn = int(np.searchsorted(obs_with, threshold, side="right"))
        tp = T_audit - fn

        tpr_low_t, fpr_up_t = get_tpr_low_fpr_up(tp, fp, T_audit)

        if fpr_up_t >= 1 or (1 - tpr_low_t) >= 1 or fpr_up_t <= 0 or (1 - tpr_low_t) <= 0:
            continue

        mu = max(norm.ppf(tpr_low_t) - norm.ppf(fpr_up_t), 0)
        if mu > max_mu:
            max_mu, best_threshold = mu, threshold

    eps_wb = gdp_to_eps(max(max_mu, 0.0), delta)
    if best_threshold is not None:
        print(f"\n[White-box] Optimal threshold={best_threshold:.6f}", file=f)
        print(f"[White-box] mu_lower={max_mu:.4f}, eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)
        eps_wb = 0.0

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ESA Active Auditing (Bootstrap + Bernstein+CP Error)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_subsets", type=int, default=8)
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=2000, help="Number of clean snapshots")
    parser.add_argument("--n_sample", type=int, default=400000, help="Bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--snapshot_gamma_frac", type=float, default=0.5,
                        help="Fraction of gamma allocated to snapshot phase")
    parser.add_argument("--replicate", type=int, default=1,
                        help="Replicate each unique snapshot this many times (e.g. 5 to generate n_llm/5 unique)")
    parser.add_argument("--y1_target", type=str, default="epigonichthys haurient todidae america tipstock stipuliform walleye zanclodontidae grapefruit dispermy")
    parser.add_argument("--y2_control", type=str, default="torchwood memorial propagandist amman just busk clunch perhaps forlornly townless")
    parser.add_argument("--output_dir", type=str, default="results_esa_active")
    parser.add_argument("--output_prefix", type=str, default="esa-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="DP flaw: calibrate noise to this sensitivity instead of the correct 2/K")
    parser.add_argument("--canary_indices", type=int, nargs="+", default=None,
                        help="Dataset row of the canary for each run; overrides --canary_mode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM call (<=1: original one-at-a-time path)")
    args = parser.parse_args()

    global generation_temperature
    generation_temperature = args.temperature

    load_hf_model(args.model_path)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(args.embedding_model)

    all_data = load_data_csv(args.data_path, encoding=DATASET_CONFIGS[args.dataset_type]["encoding"])
    if not all_data:
        print("Error: no data loaded.")
        return

    y1_embedding = embedding_model.encode(args.y1_target, convert_to_numpy=True)
    y1_embedding /= max(np.linalg.norm(y1_embedding), 1e-12)
    y2_embedding = embedding_model.encode(args.y2_control, convert_to_numpy=True)
    y2_embedding /= max(np.linalg.norm(y2_embedding), 1e-12)

    os.makedirs(args.output_dir, exist_ok=True)
    num_exemplars = args.shots_per_subset * args.num_subsets

    for run_id in range(args.num_runs):
        run_seed = None if args.seed == 0 else (args.seed + run_id)
        if run_seed is not None:
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

        rng = random.Random(run_seed) if run_seed is not None else random
        if args.canary_indices is not None:
            canary_index = args.canary_indices[run_id]
        elif args.canary_mode == "fixed":
            canary_index = 0
        else:
            canary_index = rng.randrange(len(all_data))

        canary_fact = all_data[canary_index][2]
        if args.dataset_type == "xsum":
            canary_exemplar = all_data[canary_index][2]
        else:
            canary_exemplar = all_data[canary_index][1] + " " + all_data[canary_index][2]

        sensitivity = args.sensitivity if args.sensitivity is not None else 2.0 / args.num_subsets

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        t_run = time.time()
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("ESA Active Auditing (Bootstrap + Bernstein+CP Error Bound)", file=f)
            print(f"Sensitivity: 2/K = {sensitivity:.6f}", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Estimator: GDP", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Clopper-Pearson (MC)", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, K={args.num_subsets}", file=f)
            print(f"Canary: '{canary_fact[:100]}...'", file=f)
            print("=" * 60, file=f)

            t1_start = time.time()
            vecs_with, vecs_without = collect_snapshots(
                embedding_model, all_data, canary_index, canary_exemplar,
                canary_fact, num_exemplars, args.num_subsets,
                args.y1_target, args.y2_control, args.n_llm,
                args.dataset_type, rng, f,
                replicate=args.replicate, batch_size=args.batch_size
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Stage 1: {t1_time:.1f}s", file=f)

            # Pre-compute score direction (y1 - y2, normalized)
            diff = y1_embedding - y2_embedding
            diff_norm = np.linalg.norm(diff)
            direction = diff / max(diff_norm, 1e-12)

            for epsilon in args.epsilons:
                sigma = analytic_gauss_sigma(sensitivity, epsilon, args.delta)
                # Projection of isotropic N(0,sigma^2*I) onto unit direction has std=sigma
                sigma_1d = sigma
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={sigma:.6f}, sigma_1d={sigma_1d:.6f}", file=f)

                t2_start = time.time()
                scores_with = bootstrap_simulate_gaussian_scores(
                    vecs_with, sigma_1d, args.n_sample, direction)
                scores_without = bootstrap_simulate_gaussian_scores(
                    vecs_without, sigma_1d, args.n_sample, direction)
                t2_time = time.time() - t2_start

                t3_start = time.time()
                eps_bb, eps_wb = run_active_audit(
                    scores_with, scores_without,
                    n_llm=args.n_llm, n_sample=args.n_sample,
                    delta=args.delta, gamma_conf=args.gamma_conf, f=f,
                    snapshot_gamma_frac=args.snapshot_gamma_frac
                )
                t3_time = time.time() - t3_start

                total = t1_time + t2_time + t3_time
                print(f"\n[TIMING] Stage 2: {t2_time:.1f}s, Stage 3: {t3_time:.1f}s", file=f)
                print(f"[TIMING] Total: {total:.1f}s", file=f)
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
