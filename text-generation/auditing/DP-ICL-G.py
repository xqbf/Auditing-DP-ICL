import argparse
import csv
import math
import os
import random
import sys
from typing import Any, List, Tuple, Callable

import numpy as np
import torch
from scipy.stats import beta, norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# =============================================================================
# Args
# =============================================================================

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bf16/fp16/fp32.")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "DP-ICL Combined Audit (local HF model): Mapping attack + Optimal-threshold attack.\n"
            "This script only uses text content (dialogue/summary columns). You can run other CSV datasets\n"
            "as long as you set correct column indices."
        )
    )

    # Data (CSV)
    p.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/Samsum/train.csv")
    p.add_argument("--data_encoding", type=str, default="latin-1")
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--has_header", action="store_true", default=True)
    p.add_argument("--no_header", action="store_false", dest="has_header")

    # Column indices (default assumes SAMSum-like: id, dialogue, summary)
    p.add_argument("--id_col", type=int, default=0)
    p.add_argument("--dialogue_col", type=int, default=1)
    p.add_argument("--summary_col", type=int, default=2)

    # Model paths
    p.add_argument("--local_model_path", type=str, default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF")
    p.add_argument("--embedding_model_name", type=str, default="/local/scratch/yxia327/bge-large-en-v1.5")
    p.add_argument("--embedding_device", type=str, default=None,
                   help="e.g., 'cuda' or 'cpu'. If None, SentenceTransformer decides.")

    # HF model loading
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="bf16", help="bf16|fp16|fp32")

    # Generation configs
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max_new_tokens_unified", type=int, default=100)
    p.add_argument("--max_new_tokens_white", type=int, default=400)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--no_sample", action="store_false", dest="do_sample")

    # Constrained output sentences
    p.add_argument("--y1_target", type=str, default="The old clock chimed a forgotten, dusty tune.")
    p.add_argument("--y2_control", type=str, default=(
        "FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."
    ))

    # Experiment parameters
    p.add_argument("--subsets_list", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14])
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="experiment_results_esa")
    p.add_argument("--output_prefix", type=str, default="Generation_Auditing")

    # Shot control: original code used num_exemplars = 1 * K
    p.add_argument("--shots_per_subset", type=int, default=1,
                   help="Total exemplars sampled = shots_per_subset * K. Default=1 (original behavior).")

    # Privacy/audit parameters
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--delta", type=float, default=1e-5)

    # Collection/simulation sizes
    p.add_argument("--T_collect", type=int, default=400)
    p.add_argument("--T_simulation", type=int, default=400000)
    p.add_argument("--num_candidates", type=int, default=100,
                   help="Number of 0-shot candidates to generate for mapping attack.")

    # Misc
    p.add_argument("--seed", type=int, default=0, help="0 = do not force seeding.")
    p.add_argument("--verbose", action="store_true", default=False, help="Print prompts and LLM responses.")

    return p.parse_args()


def maybe_seed(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data loading
# =============================================================================

def load_data_csv(args) -> List[Tuple[str, ...]]:
    try:
        with open(args.data_path, "r", encoding=args.data_encoding) as f:
            rows = [tuple(r) for r in csv.reader(f, delimiter=args.delimiter)]
        if args.has_header and len(rows) > 0:
            rows = rows[1:]
        all_data = rows
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at `{args.data_path}`.")

    if len(all_data) == 0:
        raise RuntimeError("Dataset is empty after loading.")

    max_idx = max(args.id_col, args.dialogue_col, args.summary_col)
    if len(all_data[0]) <= max_idx:
        raise RuntimeError(f"Row has {len(all_data[0])} cols but requested index {max_idx}.")

    return all_data


# =============================================================================
# Generators (closures over tokenizer/model/args)
# =============================================================================

def make_llm_white_generator(tokenizer, model, args) -> Callable[[List[str], str], str]:
    """Constrained generation: returns y1_target or y2_control."""
    y1_target = args.y1_target
    y2_control = args.y2_control

    def _gen(contexts: List[str], canary: str) -> str:
        exemplar_context = "\\n".join(contexts) if contexts else ""

        query_instruction = f"""

Carefully check if the dialog "{canary}" is present in the provided CONTEXTUAL EXEMPLAR:.

If "{canary}" is found in the CONTEXTUAL EXEMPLAR:, output "{y1_target}".
Otherwise, output "{y2_control}".

CONTEXTUAL EXEMPLAR:
{exemplar_context}

RESPONSE:
"""
        messages = [{"role": "user", "content": query_instruction}]
        raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens_white,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        if args.verbose:
            print(query_instruction)
            print(f"LLM Response: {response}")

        if y1_target in response:
            final_output = y1_target
        elif y2_control in response:
            final_output = y2_control
        else:
            final_output = response

        if args.verbose:
            print(f"Constrained Output: {final_output}\\n{'-'*50}\\n")

        return final_output

    return _gen


def make_llm_unified_generator(tokenizer, model, args) -> Callable[[List[str], str], str]:
    """Unified generator: context => check canary; empty context => sampling allows random output."""
    y1_target = args.y1_target
    y2_control = args.y2_control

    def _gen(contexts: List[str], canary: str) -> str:
        exemplar_context = "\\n".join(contexts) if contexts else ""

        query_instruction = f"""Analyze the "CONTEXTUAL EXEMPLAR" below. Your ONLY function is to output ONE of two predefined sentences. Do not add explanations.

1. Check if the specific phrase "{canary}" is present in the context.
2. If the context is empty, randomly output either "{y1_target}" or "{y2_control}".
3. If the context is NOT empty and "{canary}" is found, output "{y1_target}".
4. If the context is NOT empty and "{canary}" is NOT found, output "{y2_control}".

CONTEXTUAL EXEMPLAR:
{exemplar_context}

RESPONSE:
"""
        messages = [{"role": "user", "content": query_instruction}]
        raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens_unified,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        outputs = model.generate(**inputs, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        if y1_target in response:
            return y1_target
        if y2_control in response:
            return y2_control
        return response

    return _gen


def generate_zero_shot_candidates(
    num_candidates: int,
    embedding_model: Any,
    unified_generator: Callable[[List[str], str], str],
    canary_for_gen: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates candidate sentences using unified generator (empty context => random behavior)."""
    candidate_texts = [
        unified_generator(contexts=[], canary=canary_for_gen)
        for _ in tqdm(range(num_candidates), desc="Generating 0-Shot Candidates")
    ]
    unique_candidate_texts = list(set(candidate_texts))
    print(f"Generated {len(unique_candidate_texts)} unique candidate sentences.")

    candidate_embeddings = embedding_model.encode(unique_candidate_texts, convert_to_numpy=True)
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    normalized_embeddings = candidate_embeddings / np.where(norms == 0, 1, norms)

    return normalized_embeddings, np.array(unique_candidate_texts)


# =============================================================================
# DP-ICL (ESA) mechanism
# =============================================================================

class DPICLForESA:
    def __init__(self, llm_callable, embedding_model, epsilon: float, delta: float, embedding_dim: int):
        self.llm = llm_callable
        self.embedding_model = embedding_model
        self.epsilon = epsilon
        self.delta = delta
        self.embedding_dim = embedding_dim
        self.sigma = 0.0

    def _get_subset_embedding(self, subset_exemplar_str: str, canary: str) -> np.ndarray:
        output_sentence = self.llm(contexts=[subset_exemplar_str], canary=canary)
        embedding = self.embedding_model.encode(output_sentence, convert_to_numpy=True)
        norm_val = np.linalg.norm(embedding)
        return embedding / norm_val if norm_val > 0 else np.zeros_like(embedding)

    def get_clean_mean_embedding(self, exemplars_list: List[str], num_subsets: int, canary: str) -> np.ndarray:
        shuffled_indices = np.random.permutation(len(exemplars_list))
        partition_indices = np.array_split(shuffled_indices, num_subsets)

        sum_embedding = np.zeros(self.embedding_dim)
        for subset_indices in partition_indices:
            if len(subset_indices) == 0:
                continue
            subset_exemplar_str = exemplars_list[subset_indices[0]]  # original behavior
            sum_embedding += self._get_subset_embedding(subset_exemplar_str, canary)

        return sum_embedding / num_subsets if num_subsets > 0 else np.zeros(self.embedding_dim)


# =============================================================================
# Audit helpers
# =============================================================================

def clopper_pearson(successes, trials, confidence_level):
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)


def gdp_to_eps(mu, delta):
    if mu <= 0:
        return 0.0
    min_eps, max_eps = 0, 100
    for _ in range(100):
        eps = (min_eps + max_eps) / 2
        term1 = norm.cdf(-eps / mu + mu / 2)
        term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        if (term1 - term2) > delta:
            min_eps = eps
        else:
            max_eps = eps
    return min_eps


def collect_clean_mean_embeddings_for_resampling(dp_esa, exemplars_with, exemplars_without, canary, num_subsets, T_collect):
    print(f"\\n--- STAGE 1: Collecting {T_collect} clean mean embeddings ---")
    vectors_without = [
        dp_esa.get_clean_mean_embedding(exemplars_without, num_subsets, canary)
        for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")
    ]
    vectors_with = [
        dp_esa.get_clean_mean_embedding(exemplars_with, num_subsets, canary)
        for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")
    ]
    return vectors_with, vectors_without


def simulate_noisy_embeddings_from_bootstrap(collected_vectors, sigma, T_simulation):
    resampled_indices = np.random.choice(len(collected_vectors), T_simulation, replace=True)
    resampled_vectors = np.array(collected_vectors)[resampled_indices]
    noise = np.random.normal(0, sigma, resampled_vectors.shape)
    return resampled_vectors + noise


# =============================================================================
# Audit 1: mapping attack
# =============================================================================

def select_final_answer_and_get_embedding(noisy_embeds, candidate_embeddings):
    similarities = np.dot(noisy_embeds, candidate_embeddings.T)
    best_indices = np.argmax(similarities, axis=1)
    return candidate_embeddings[best_indices]


def run_mapping_attack(dp_esa, noisy_embeds_with, noisy_embeds_without, E_in, E_out, f, candidate_embeddings):
    print("\\n--- AUDIT 1: Running 0-Shot Mapping Attack ---", file=f)

    final_embeds_with = select_final_answer_and_get_embedding(noisy_embeds_with, candidate_embeddings)
    final_embeds_without = select_final_answer_and_get_embedding(noisy_embeds_without, candidate_embeddings)

    T_audit = len(final_embeds_with)
    dist_sq_to_in_w = np.sum((final_embeds_with - E_in) ** 2, axis=1)
    dist_sq_to_out_w = np.sum((final_embeds_with - E_out) ** 2, axis=1)
    tp_count = int(np.sum(dist_sq_to_in_w < dist_sq_to_out_w))

    dist_sq_to_in_wo = np.sum((final_embeds_without - E_in) ** 2, axis=1)
    dist_sq_to_out_wo = np.sum((final_embeds_without - E_out) ** 2, axis=1)
    fp_count = int(np.sum(dist_sq_to_in_wo < dist_sq_to_out_wo))

    fn_count = T_audit - tp_count
    tpr = tp_count / T_audit if T_audit > 0 else 0.0
    fpr = fp_count / T_audit if T_audit > 0 else 0.0

    print(f"TPR = {tpr:.4f} ({tp_count}/{T_audit}) | FPR = {fpr:.4f} ({fp_count}/{T_audit})", file=f)

    _, alpha_upper = clopper_pearson(fp_count, T_audit, 0.95)
    _, beta_upper = clopper_pearson(fn_count, T_audit, 0.95)
    mu_lower_bound = max(0, norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper))
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_esa.delta)

    print(f"Result (Mapping Attack): μ_lower={mu_lower_bound:.4f}, ε_lower={eps_lower_bound:.4f}", file=f)
    return float(eps_lower_bound)


# =============================================================================
# Audit 2: optimal threshold
# =============================================================================

def run_optimal_threshold_attack(dp_esa, noisy_embeds_with, noisy_embeds_without, y1_embedding, y2_embedding, f):
    print("\\n--- AUDIT 2: Running Optimal Threshold Attack ---", file=f)

    scores_with = np.dot(noisy_embeds_with, y1_embedding) - np.dot(noisy_embeds_with, y2_embedding)
    scores_without = np.dot(noisy_embeds_without, y1_embedding) - np.dot(noisy_embeds_without, y2_embedding)

    all_scores = np.concatenate((scores_with, scores_without))
    labels = np.concatenate((np.ones_like(scores_with), np.zeros_like(scores_without)))

    sorted_indices = np.argsort(all_scores)
    sorted_labels = labels[sorted_indices]

    max_mu, best_tpr, best_fpr = -1.0, 0.0, 0.0
    tp, fp = len(scores_with), len(scores_without)

    for i in range(len(all_scores)):
        if sorted_labels[i] == 1:
            tp -= 1
        else:
            fp -= 1

        tpr = tp / len(scores_with) if len(scores_with) > 0 else 0.0
        fpr = fp / len(scores_without) if len(scores_without) > 0 else 0.0
        if tpr < fpr:
            continue

        _, alpha_upper = clopper_pearson(int(fpr * len(scores_without)), len(scores_without), 0.95)
        _, beta_upper = clopper_pearson(int((1 - tpr) * len(scores_with)), len(scores_with), 0.95)
        mu_lower = max(0, norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper))

        if mu_lower > max_mu:
            max_mu, best_tpr, best_fpr = mu_lower, tpr, fpr

    print(f"Best TPR = {best_tpr:.4f} | Best FPR = {best_fpr:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(max_mu, dp_esa.delta)
    print(f"Result (Optimal Attack): μ_lower={max_mu:.4f}, ε_lower={eps_lower_bound:.4f}", file=f)
    return float(eps_lower_bound)


# =============================================================================
# Procedure
# =============================================================================

def procedure(output_filename, num_subsets, all_data, embedding_model, embedding_dim,
              unified_generator, white_generator, args):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        print("=" * 20 + f" DP-ICL Combined Audit (K={num_subsets}) " + "=" * 20, file=f)

        if args.shots_per_subset < 1:
            print("Error: --shots_per_subset must be >= 1", file=f)
            return {}

        # 1) Data preparation
        num_exemplars = args.shots_per_subset * num_subsets
        total_samples = num_exemplars + 1
        if total_samples > len(all_data):
            print("Warning: Not enough data.", file=f)
            return {}

        selected_indices = random.sample(range(len(all_data)), total_samples)
        canary_index, base_indices = selected_indices[-1], selected_indices[:-1]

        canary_fact = all_data[canary_index][args.summary_col]
        print(f"Selected Canary Fact: {canary_fact}\\n", file=f)

        exemplars_without = [
            all_data[i][args.dialogue_col] + " " + all_data[i][args.summary_col]
            for i in base_indices
        ]
        exemplars_with = list(exemplars_without)
        exemplars_with[random.randint(0, len(exemplars_with) - 1)] = (
            all_data[canary_index][args.dialogue_col] + " " + all_data[canary_index][args.summary_col]
        )

        # 2) Setup for attacks
        candidate_embeddings, _ = generate_zero_shot_candidates(
            args.num_candidates, embedding_model, unified_generator, canary_fact
        )
        y1_embedding = embedding_model.encode(args.y1_target, convert_to_numpy=True)
        y1_embedding /= max(np.linalg.norm(y1_embedding), 1e-12)
        y2_embedding = embedding_model.encode(args.y2_control, convert_to_numpy=True)
        y2_embedding /= max(np.linalg.norm(y2_embedding), 1e-12)

        # 3) Collect clean empirical distributions
        collector_esa = DPICLForESA(unified_generator, embedding_model, 999.0, args.delta, embedding_dim)
        vecs_with, vecs_without = collect_clean_mean_embeddings_for_resampling(
            collector_esa, exemplars_with, exemplars_without, canary_fact, num_subsets, args.T_collect
        )
        E_true_in, E_true_out = np.mean(vecs_with, axis=0), np.mean(vecs_without, axis=0)

        results = {}
        for epsilon in args.epsilons:
            dp_esa = DPICLForESA(unified_generator, embedding_model, epsilon, args.delta, embedding_dim)
            dp_esa_white = DPICLForESA(white_generator, embedding_model, epsilon, args.delta, embedding_dim)

            sensitivity = 2.0 / num_subsets
            sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / dp_esa.delta))) / max(dp_esa.epsilon, 1e-12)
            dp_esa.sigma = sigma
            dp_esa_white.sigma = sigma

            print(f"\\n--- CONFIG: K={num_subsets}, ε={dp_esa.epsilon}, σ={dp_esa.sigma:.4f} ---", file=f)

            noisy_with = simulate_noisy_embeddings_from_bootstrap(vecs_with, dp_esa.sigma, args.T_simulation)
            noisy_without = simulate_noisy_embeddings_from_bootstrap(vecs_without, dp_esa.sigma, args.T_simulation)

            eps_map = run_mapping_attack(dp_esa, noisy_with, noisy_without, E_true_in, E_true_out, f, candidate_embeddings)
            eps_opt = run_optimal_threshold_attack(dp_esa_white, noisy_with, noisy_without, y1_embedding, y2_embedding, f)

            results[epsilon] = {"mapping_attack_eps": eps_map, "optimal_attack_eps": eps_opt}

        print(f"\\nAudit finished for K={num_subsets}. Results saved to '{output_filename}'")
        return results


def main():
    args = parse_args()
    maybe_seed(args.seed)

    # Load dataset
    all_data = load_data_csv(args)

    # Load tokenizer/model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)

    print("Loading LLM model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_path,
        device_map=args.device_map,
        torch_dtype=_torch_dtype_from_str(args.torch_dtype),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load embedding model
    print("Loading Embedding model...")
    if args.embedding_device is None:
        embedding_model = SentenceTransformer(args.embedding_model_name)
    else:
        embedding_model = SentenceTransformer(args.embedding_model_name, device=args.embedding_device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    # Generators
    white_generator = make_llm_white_generator(tokenizer, model, args)
    unified_generator = make_llm_unified_generator(tokenizer, model, args)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for num_subsets in args.subsets_list:
        for run_id in range(1, args.num_runs + 1):
            if args.seed and args.seed > 0:
                maybe_seed(args.seed + run_id + 1000 * num_subsets)

            output_filename = os.path.join(
                args.output_dir,
                f"{args.output_prefix}{run_id}-subsets{num_subsets}.txt",
            )
            print(f"\\nStarting Run {run_id}/{args.num_runs} for K={num_subsets} subsets...")
            all_results[(run_id, num_subsets)] = procedure(
                output_filename, num_subsets, all_data, embedding_model, embedding_dim,
                unified_generator, white_generator, args
            )

    print("\\n\\n" + "=" * 30 + "\\nFinal Summary of All Runs\\n" + "=" * 30)
    for (run_id, num_subsets), res in all_results.items():
        if not res:
            continue
        print(f"Run {run_id}, Subsets {num_subsets}:")
        for epsilon, values in res.items():
            print(
                f"  ε_theory={epsilon:.1f} -> ε_map={values['mapping_attack_eps']:.4f}, " \
                f"ε_opt={values['optimal_attack_eps']:.4f}"
            )


if __name__ == "__main__":
    main()
