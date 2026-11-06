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
    batch_size <= 1: plain one-at-a-time calls, no cache (for timing baselines)."""
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


class ESAClassifier:
    def __init__(self, embedding_model, epsilon, delta, num_subsets, sensitivity=None):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        self.epsilon = epsilon
        self.delta = delta
        self.num_subsets = num_subsets
        self.sensitivity = sensitivity if sensitivity is not None else 2.0 / num_subsets
        self.sigma = analytic_gauss_sigma(self.sensitivity, epsilon, delta)

    def get_subset_embedding(self, context_text, canary, y1_target, y2_control):
        output = generate_signal_sentence([context_text], canary, y1_target, y2_control)
        embedding = self.embedding_model.encode(output, convert_to_numpy=True)
        norm_val = np.linalg.norm(embedding)
        return embedding / norm_val if norm_val > 0 else np.zeros_like(embedding)

    def run_full_processes(self, exemplar_lists, canary, y1_target, y2_control, batch_size,
                           desc=None):
        """Full ESA pipeline for every trial: partition → embed → mean → Gaussian noise.
        Every trial's partition is drawn first, all subset generations (and their
        embeddings) run in batched passes, then each trial adds its own noise.
        batch_size <= 1 keeps one LLM call and one embedding call per subset."""
        trial_contexts = []
        for exemplars in exemplar_lists:
            shuffled = np.random.permutation(len(exemplars))
            trial_contexts.append([exemplars[s[0]] for s in np.array_split(shuffled, self.num_subsets)
                                   if len(s) > 0])
        outputs = generate_signal_sentences([c for cs in trial_contexts for c in cs],
                                            canary, y1_target, y2_control, batch_size, desc)
        if batch_size <= 1:
            embs = np.array([self.embedding_model.encode(o, convert_to_numpy=True) for o in outputs])
        else:
            embs = self.embedding_model.encode(outputs, convert_to_numpy=True, batch_size=batch_size)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / np.where(norms > 0, norms, 1.0), 0.0)

        results, pos = [], 0
        for cs in trial_contexts:
            mean_emb = embs[pos:pos + len(cs)].sum(axis=0) / self.num_subsets
            pos += len(cs)
            noise = np.random.normal(0, self.sigma, self.embedding_dim)
            results.append((mean_emb + noise, mean_emb))
        return results


# ==============================================================================
# Passive Audit
# ==============================================================================
def run_passive_audit(esa, all_data, canary_index, canary_exemplar, canary_fact,
                      num_exemplars, y1_target, y2_control,
                      y1_embedding, y2_embedding,
                      n_trials, delta, dataset_type, rng, f, batch_size):
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

    # Draw all trial contexts first, batch the LLM + embeddings, then score per trial.
    trials_with = esa.run_full_processes(
        [sample_exemplars_with() for _ in range(n_trials)], canary_fact, y1_target, y2_control,
        batch_size, desc="Passive trials (with canary)")
    trials_without = esa.run_full_processes(
        [sample_exemplars_without() for _ in range(n_trials)], canary_fact, y1_target, y2_control,
        batch_size, desc="Passive trials (without canary)")

    for noisy_emb, _ in trials_with:
        sim_y1 = np.dot(noisy_emb, y1_embedding)
        sim_y2 = np.dot(noisy_emb, y2_embedding)
        if sim_y1 > sim_y2:
            bb_tp += 1
        scores_with.append(sim_y1 - sim_y2)

    for noisy_emb, _ in trials_without:
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
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="DP flaw: calibrate noise to this sensitivity instead of the correct 2/K")
    parser.add_argument("--canary_indices", type=int, nargs="+", default=None,
                        help="Dataset row of the canary for each run; overrides --canary_mode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0 = greedy decoding)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Prompts per batched LLM generate call (1 = one call at a time)")
    parser.add_argument("--seed", type=int, default=0)
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

        output_file = os.path.join(args.output_dir, f"{args.output_prefix}-run{run_id}.txt")
        t_run = time.time()
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
                esa = ESAClassifier(embedding_model, epsilon, args.delta, args.num_subsets, args.sensitivity)
                print(f"\n{'='*40} eps_theory={epsilon} {'='*40}", file=f)
                print(f"sigma={esa.sigma:.6f}", file=f)

                t0 = time.time()
                eps_bb, eps_wb = run_passive_audit(
                    esa, all_data, canary_index, canary_exemplar, canary_fact,
                    num_exemplars, args.y1_target, args.y2_control,
                    y1_embedding, y2_embedding,
                    args.n_trials, args.delta, args.dataset_type, rng, f, args.batch_size
                )
                elapsed = time.time() - t0

                print(f"\n[RESULT] eps_theory={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}", file=f)
                print(f"[TIMING] {elapsed:.1f}s total", file=f)
                print(f"Run {run_id}, eps={epsilon}: bb={eps_bb:.4f}, wb={eps_wb:.4f}")

        wall = time.time() - t_run
        with open(output_file, "a") as f:
            print(f"\n[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons "
                  f"(LLM + audit, excl. model load)", file=f)
        print(f"[WALLCLOCK] run {run_id}: {wall:.1f}s for {len(args.epsilons)} epsilons")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
