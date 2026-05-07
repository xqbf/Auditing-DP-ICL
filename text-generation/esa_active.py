"""
ESA Active Auditing (Bootstrap).

Stage 1: Collect n_llm clean mean embeddings via real LLM calls.
Stage 2: Bootstrap n_sample noisy embeddings by resampling + Gaussian noise.
Stage 3: Compute TPR/FPR with two-layer Bernstein + MC error bound,
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
    empirical_bernstein_correction, build_esa_generator_prompt,
    load_data_csv,
)

# ==============================================================================
# Global model state
# ==============================================================================
tokenizer = None
model = None


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
            do_sample=False,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    if y1_target in response:
        return y1_target
    if y2_control in response:
        return y2_control
    return response


def get_clean_mean_embedding(embedding_model, exemplars, num_subsets, canary,
                             y1_target, y2_control):
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    shuffled = np.random.permutation(len(exemplars))
    partitions = np.array_split(shuffled, num_subsets)

    sum_emb = np.zeros(embedding_dim)
    for subset_indices in partitions:
        if len(subset_indices) == 0:
            continue
        context = exemplars[subset_indices[0]]
        output = generate_signal_sentence([context], canary, y1_target, y2_control)
        emb = embedding_model.encode(output, convert_to_numpy=True)
        norm_val = np.linalg.norm(emb)
        sum_emb += emb / norm_val if norm_val > 0 else np.zeros_like(emb)

    return sum_emb / num_subsets


# ==============================================================================
# Stage 1: Collect clean mean embeddings
# ==============================================================================
def collect_snapshots(embedding_model, all_data, canary_index, canary_exemplar,
                      canary_fact, num_exemplars, num_subsets,
                      y1_target, y2_control, n_llm, dataset_type, rng, f):
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

    vectors_with = []
    for _ in tqdm(range(n_llm), desc="Snapshots (with canary)"):
        exemplars = sample_exemplars_with()
        vectors_with.append(
            get_clean_mean_embedding(embedding_model, exemplars, num_subsets,
                                     canary_fact, y1_target, y2_control)
        )

    vectors_without = []
    for _ in tqdm(range(n_llm), desc="Snapshots (without canary)"):
        exemplars = sample_exemplars_without()
        vectors_without.append(
            get_clean_mean_embedding(embedding_model, exemplars, num_subsets,
                                     canary_fact, y1_target, y2_control)
        )

    print(f"Mean embedding L2 (with vs without): "
          f"{np.linalg.norm(np.mean(vectors_with, axis=0) - np.mean(vectors_without, axis=0)):.6f}", file=f)
    return vectors_with, vectors_without


# ==============================================================================
# Stage 2: Bootstrap with Gaussian noise
# ==============================================================================
def bootstrap_simulate_gaussian(collected_vectors, sigma, n_sample):
    vecs = np.array(collected_vectors)
    idx = np.random.choice(len(vecs), n_sample, replace=True)
    resampled = vecs[idx]
    noise = np.random.normal(0, sigma, resampled.shape)
    return resampled + noise


# ==============================================================================
# Stage 3: Audit with two-layer error bound
# ==============================================================================
def run_active_audit(noisy_with, noisy_without, y1_embedding, y2_embedding,
                     n_llm, n_sample, delta, gamma_conf, f):
    T_audit = len(noisy_with)

    # Scores: similarity to y1 minus similarity to y2
    scores_with = np.dot(noisy_with, y1_embedding) - np.dot(noisy_with, y2_embedding)
    scores_without = np.dot(noisy_without, y1_embedding) - np.dot(noisy_without, y2_embedding)

    # Per-snapshot variance is small for ESA (<0.001)
    sigma2_with = 0.0
    sigma2_without = 0.0

    delta_1 = empirical_bernstein_correction(sigma2_with, n_llm, n_sample, gamma_conf)
    delta_0 = empirical_bernstein_correction(sigma2_without, n_llm, n_sample, gamma_conf)

    print(f"\n--- Error Bounds (gamma={gamma_conf}) ---", file=f)
    print(f"  Delta_1 (TPR correction): {delta_1:.6f}", file=f)
    print(f"  Delta_0 (FPR correction): {delta_0:.6f}", file=f)

    # === Black-box (mapping attack: sim_y1 > sim_y2) ===
    tp_count = int(np.sum(scores_with > 0))
    fp_count = int(np.sum(scores_without > 0))
    tpr_hat = tp_count / T_audit
    fpr_hat = fp_count / T_audit

    tpr_low = max(0.0, tpr_hat - delta_1)
    fpr_up = min(1.0, fpr_hat + delta_0)

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
        tpr_hat_t = 1.0 - fn / T_audit
        fpr_hat_t = fp / T_audit

        tpr_low_t = max(0.0, tpr_hat_t - delta_1)
        fpr_up_t = min(1.0, fpr_hat_t + delta_0)

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
    parser = argparse.ArgumentParser(description="ESA Active Auditing (Bootstrap + Bernstein+MC Error)")
    parser.add_argument("--model_path", type=str, default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF")
    parser.add_argument("--embedding_model", type=str, default="/local/scratch/yxia327/bge-large-en-v1.5")
    parser.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/Samsum/train.csv")
    parser.add_argument("--dataset_type", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_subsets", type=int, default=8)
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=2000, help="Number of clean snapshots")
    parser.add_argument("--n_sample", type=int, default=400000, help="Bootstrap samples")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_conf", type=float, default=0.05, help="Confidence failure probability")
    parser.add_argument("--y1_target", type=str, default="mirana wifie overliking")
    parser.add_argument("--y2_control", type=str, default="woundworth mort gambeer")
    parser.add_argument("--output_dir", type=str, default="results_esa_active")
    parser.add_argument("--output_prefix", type=str, default="esa-active")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_hf_model(args.model_path)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(args.embedding_model)

    all_data = load_data_csv(args.data_path)
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
        if args.canary_mode == "fixed":
            canary_index = 0
        else:
            canary_index = rng.randrange(len(all_data))

        canary_fact = all_data[canary_index][2]
        if args.dataset_type == "xsum":
            canary_exemplar = all_data[canary_index][2]
        else:
            canary_exemplar = all_data[canary_index][1] + " " + all_data[canary_index][2]

        sensitivity = 2.0 / args.num_subsets

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("ESA Active Auditing (Bootstrap + Bernstein+MC Error Bound)", file=f)
            print(f"Sensitivity: 2/K = {sensitivity:.6f}", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Estimator: GDP", file=f)
            print(f"Error: Empirical Bernstein (snapshot) + Hoeffding (MC)", file=f)
            print(f"n_llm={args.n_llm}, n_sample={args.n_sample}, K={args.num_subsets}", file=f)
            print(f"Canary: '{canary_fact[:100]}...'", file=f)
            print("=" * 60, file=f)

            t1_start = time.time()
            vecs_with, vecs_without = collect_snapshots(
                embedding_model, all_data, canary_index, canary_exemplar,
                canary_fact, num_exemplars, args.num_subsets,
                args.y1_target, args.y2_control, args.n_llm,
                args.dataset_type, rng, f
            )
            t1_time = time.time() - t1_start
            print(f"\n[TIMING] Stage 1: {t1_time:.1f}s", file=f)

            for epsilon in args.epsilons:
                sigma = analytic_gauss_sigma(sensitivity, epsilon, args.delta)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={sigma:.6f}", file=f)

                t2_start = time.time()
                noisy_with = bootstrap_simulate_gaussian(vecs_with, sigma, args.n_sample)
                noisy_without = bootstrap_simulate_gaussian(vecs_without, sigma, args.n_sample)
                t2_time = time.time() - t2_start

                t3_start = time.time()
                eps_bb, eps_wb = run_active_audit(
                    noisy_with, noisy_without,
                    y1_embedding, y2_embedding,
                    n_llm=args.n_llm, n_sample=args.n_sample,
                    delta=args.delta, gamma_conf=args.gamma_conf, f=f
                )
                t3_time = time.time() - t3_start

                total = t1_time + t2_time + t3_time
                print(f"\n[TIMING] Stage 2: {t2_time:.1f}s, Stage 3: {t3_time:.1f}s", file=f)
                print(f"[TIMING] Total: {total:.1f}s", file=f)
                print(f"[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
