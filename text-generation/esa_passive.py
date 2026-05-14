"""
ESA Passive Auditing (Full Process).

Runs the full Embedding Space Aggregation pipeline end-to-end for each trial:
  1. Each subset generates a signal sentence → embedding → normalize
  2. Mean aggregation across subsets
  3. Gaussian noise (analytic calibration, sensitivity=2/K)
  4. Map to nearest candidate sentence

Uses Clopper-Pearson CI. Uses GDP estimator (Gaussian mechanism).

Usage:
    python esa_passive.py --data_path /path/to/samsum/train.csv
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
# ESA Mechanism
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


class ESAClassifier:
    def __init__(self, embedding_model, epsilon, delta, num_subsets):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.epsilon = epsilon
        self.delta = delta
        self.num_subsets = num_subsets
        self.sensitivity = 2.0 / num_subsets
        self.sigma = analytic_gauss_sigma(self.sensitivity, epsilon, delta)

    def get_subset_embedding(self, context_text, canary, y1_target, y2_control):
        output = generate_signal_sentence([context_text], canary, y1_target, y2_control)
        embedding = self.embedding_model.encode(output, convert_to_numpy=True)
        norm_val = np.linalg.norm(embedding)
        return embedding / norm_val if norm_val > 0 else np.zeros_like(embedding)

    def run_full_process(self, exemplars, canary, y1_target, y2_control):
        """Full ESA pipeline: partition → embed → mean → Gaussian noise."""
        shuffled = np.random.permutation(len(exemplars))
        partitions = np.array_split(shuffled, self.num_subsets)

        sum_emb = np.zeros(self.embedding_dim)
        for subset_indices in partitions:
            if len(subset_indices) == 0:
                continue
            context = exemplars[subset_indices[0]]
            sum_emb += self.get_subset_embedding(context, canary, y1_target, y2_control)

        mean_emb = sum_emb / self.num_subsets
        noise = np.random.normal(0, self.sigma, self.embedding_dim)
        noisy_emb = mean_emb + noise
        return noisy_emb, mean_emb


# ==============================================================================
# Passive Audit
# ==============================================================================
def run_passive_audit(esa, all_data, canary_index, canary_exemplar, canary_fact,
                      num_exemplars, y1_target, y2_control,
                      y1_embedding, y2_embedding,
                      n_trials, delta, dataset_type, rng, f):
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

    # Black-box: mapping attack (closest to y1 or y2)
    bb_tp, bb_fp = 0, 0
    # White-box: continuous score (similarity to y1 - similarity to y2)
    scores_with, scores_without = [], []

    for _ in tqdm(range(n_trials), desc="Passive trials (with canary)"):
        exemplars = sample_exemplars_with()
        noisy_emb, _ = esa.run_full_process(exemplars, canary_fact, y1_target, y2_control)
        sim_y1 = np.dot(noisy_emb, y1_embedding)
        sim_y2 = np.dot(noisy_emb, y2_embedding)
        if sim_y1 > sim_y2:
            bb_tp += 1
        scores_with.append(sim_y1 - sim_y2)

    for _ in tqdm(range(n_trials), desc="Passive trials (without canary)"):
        exemplars = sample_exemplars_without()
        noisy_emb, _ = esa.run_full_process(exemplars, canary_fact, y1_target, y2_control)
        sim_y1 = np.dot(noisy_emb, y1_embedding)
        sim_y2 = np.dot(noisy_emb, y2_embedding)
        if sim_y1 > sim_y2:
            bb_fp += 1
        scores_without.append(sim_y1 - sim_y2)

    confidence = 0.95

    # === Black-box (mapping attack) ===
    bb_fn = n_trials - bb_tp
    _, alpha_upper_bb = clopper_pearson(bb_fp, n_trials, confidence)
    _, beta_upper_bb = clopper_pearson(bb_fn, n_trials, confidence)
    mu_bb = max(0, norm.ppf(1 - beta_upper_bb) - norm.ppf(alpha_upper_bb))
    eps_bb = gdp_to_eps(mu_bb, delta)
    print(f"\n[Black-box] TPR={bb_tp/n_trials:.4f}, FPR={bb_fp/n_trials:.4f}", file=f)
    print(f"[Black-box] mu_lower={mu_bb:.4f}, eps_lower={eps_bb:.4f}", file=f)

    # === White-box (optimal threshold sweep) ===
    obs_with = np.sort(np.array(scores_with))
    obs_without = np.sort(np.array(scores_without))
    candidates = np.unique(np.concatenate((obs_with, obs_without)))

    max_mu = -float("inf")
    best_threshold = None

    for threshold in candidates:
        fp = n_trials - int(np.searchsorted(obs_without, threshold, side="right"))
        fn = int(np.searchsorted(obs_with, threshold, side="right"))
        _, au = clopper_pearson(fp, n_trials, confidence)
        _, bu = clopper_pearson(fn, n_trials, confidence)
        if au >= 1 or bu >= 1 or au <= 0 or bu <= 0:
            continue
        mu = max(0, norm.ppf(1 - bu) - norm.ppf(au))
        if mu > max_mu:
            max_mu, best_threshold = mu, threshold

    eps_wb = gdp_to_eps(max(max_mu, 0.0), delta)
    if best_threshold is not None:
        print(f"[White-box] Optimal threshold={best_threshold:.6f}", file=f)
        print(f"[White-box] mu_lower={max_mu:.4f}, eps_lower={eps_wb:.4f}", file=f)
    else:
        print("[White-box] Could not find a valid threshold.", file=f)
        eps_wb = 0.0

    return eps_bb, eps_wb


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ESA Passive Auditing (Full Process)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="samsum", choices=["samsum", "xsum"])
    parser.add_argument("--num_subsets", type=int, default=8)
    parser.add_argument("--shots_per_subset", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=20000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--y1_target", type=str, default="mirana wifie overliking")
    parser.add_argument("--y2_control", type=str, default="woundworth mort gambeer")
    parser.add_argument("--output_dir", type=str, default="results_esa_passive")
    parser.add_argument("--output_prefix", type=str, default="esa-passive")
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

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        with open(output_file, "w") as f:
            print("=" * 60, file=f)
            print("ESA Passive Auditing (Full Process)", file=f)
            print(f"Sensitivity: 2/K = {2.0/args.num_subsets:.6f}", file=f)
            print(f"Noise: Analytic Gaussian (Balle & Wang 2018)", file=f)
            print(f"Estimator: GDP", file=f)
            print(f"Confidence: Clopper-Pearson 95%", file=f)
            print(f"n_trials={args.n_trials}, K={args.num_subsets}", file=f)
            print(f"Canary: '{canary_fact[:100]}...'", file=f)
            print("=" * 60, file=f)

            for epsilon in args.epsilons:
                esa = ESAClassifier(embedding_model, epsilon, args.delta, args.num_subsets)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={esa.sigma:.6f}", file=f)

                t0 = time.time()
                eps_bb, eps_wb = run_passive_audit(
                    esa, all_data, canary_index, canary_exemplar, canary_fact,
                    num_exemplars, args.y1_target, args.y2_control,
                    y1_embedding, y2_embedding,
                    args.n_trials, args.delta, args.dataset_type, rng, f
                )
                elapsed = time.time() - t0

                print(f"\n[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"[TIMING] {elapsed:.1f}s total", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
