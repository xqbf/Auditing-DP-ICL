"""
DP Flaw: ESA with Wrong Sensitivity.

Demonstrates that using sensitivity=1.0 (the original DP-ICL paper's claim)
instead of the correct sensitivity=2/K leads to insufficient noise,
causing empirical epsilon to exceed theoretical epsilon.

The correct sensitivity for ESA is 2/K: each subset produces a unit-norm
embedding, so swapping one data point changes one subset's embedding by at
most 2 in L2, and the mean changes by at most 2/K.

This script runs active auditing (bootstrap) with sensitivity=1.0 (WRONG)
to show the privacy violation.

Usage:
    python esa_flaw_wrong_sensitivity.py --epsilons 1 2 4 8 16
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
    build_esa_generator_prompt, load_data_csv,
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
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DP Flaw: ESA Wrong Sensitivity")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_subsets", type=int, default=8)
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_llm", type=int, default=20000)
    parser.add_argument("--n_sample", type=int, default=10000000, help="10M bootstrap for precise flaw detection")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--gamma_conf", type=float, default=0.05)
    parser.add_argument("--snapshot_gamma_frac", type=float, default=0.5)
    parser.add_argument("--y1_target", type=str, default="mirana wifie overliking")
    parser.add_argument("--y2_control", type=str, default="woundworth mort gambeer")
    parser.add_argument("--output_dir", type=str, default="results_esa_flaw")
    parser.add_argument("--output_prefix", type=str, default="esa-flaw-wrong-sens")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canary_mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    args = parser.parse_args()

    global generation_temperature
    generation_temperature = args.temperature

    load_hf_model(args.model_path)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(args.embedding_model)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

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

        remaining = [i for i in range(len(all_data)) if i != canary_index]

        def sample_with():
            indices = rng.sample(remaining, num_exemplars - 1)
            if args.dataset_type == "xsum":
                base = [all_data[i][2] for i in indices]
            else:
                base = [all_data[i][1] + " " + all_data[i][2] for i in indices]
            pos = random.randint(0, len(base))
            base.insert(pos, canary_exemplar)
            return base

        def sample_without():
            indices = rng.sample(remaining, num_exemplars)
            if args.dataset_type == "xsum":
                return [all_data[i][2] for i in indices]
            return [all_data[i][1] + " " + all_data[i][2] for i in indices]

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("DP FLAW: ESA with Wrong Sensitivity (sens=1.0 instead of 2/K)", file=f)
            print("Expected: eps_empirical > eps_theory at various epsilon", file=f)
            print(f"Correct sensitivity: 2/K = {2.0/args.num_subsets:.6f}", file=f)
            print(f"Wrong sensitivity used: 1.0", file=f)
            print(f"Canary: '{canary_fact[:100]}...'", file=f)
            print("=" * 60, file=f)

            # Stage 1: collect clean snapshots
            vecs_with, vecs_without = [], []
            for _ in tqdm(range(args.n_llm), desc="Snapshots (with)"):
                exemplars = sample_with()
                vecs_with.append(
                    get_clean_mean_embedding(embedding_model, exemplars, args.num_subsets,
                                             canary_fact, args.y1_target, args.y2_control)
                )
            for _ in tqdm(range(args.n_llm), desc="Snapshots (without)"):
                exemplars = sample_without()
                vecs_without.append(
                    get_clean_mean_embedding(embedding_model, exemplars, args.num_subsets,
                                             canary_fact, args.y1_target, args.y2_control)
                )

            print(f"Mean embedding L2: {np.linalg.norm(np.mean(vecs_with, axis=0) - np.mean(vecs_without, axis=0)):.6f}", file=f)

            for epsilon in args.epsilons:
                # WRONG sensitivity = 1.0 (the original paper's claim)
                wrong_sensitivity = 1.0
                sigma_wrong = analytic_gauss_sigma(wrong_sensitivity, epsilon, args.delta)

                # Correct sensitivity for comparison
                correct_sensitivity = 2.0 / args.num_subsets
                sigma_correct = analytic_gauss_sigma(correct_sensitivity, epsilon, args.delta)

                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma_wrong (sens=1.0)={sigma_wrong:.6f}", file=f)
                print(f"sigma_correct (sens=2/K)={sigma_correct:.6f}", file=f)

                # Bootstrap with WRONG sigma
                vw = np.array(vecs_with)
                vwo = np.array(vecs_without)
                idx_w = np.random.choice(len(vw), args.n_sample, replace=True)
                idx_wo = np.random.choice(len(vwo), args.n_sample, replace=True)
                noisy_w = vw[idx_w] + np.random.normal(0, sigma_wrong, (args.n_sample, embedding_dim))
                noisy_wo = vwo[idx_wo] + np.random.normal(0, sigma_wrong, (args.n_sample, embedding_dim))

                # Scores: similarity to y1 - similarity to y2
                scores_w = np.dot(noisy_w, y1_embedding) - np.dot(noisy_w, y2_embedding)
                scores_wo = np.dot(noisy_wo, y1_embedding) - np.dot(noisy_wo, y2_embedding)

                tp_count = int(np.sum(scores_w > 0))
                fp_count = int(np.sum(scores_wo > 0))
                tpr_hat = tp_count / args.n_sample
                fpr_hat = fp_count / args.n_sample

                # Bernstein snapshot correction + Clopper-Pearson for MC
                gamma_snap = args.gamma_conf * args.snapshot_gamma_frac
                gamma_mc = args.gamma_conf * (1.0 - args.snapshot_gamma_frac)
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
