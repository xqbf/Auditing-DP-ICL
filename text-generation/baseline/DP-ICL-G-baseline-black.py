import numpy as np
import csv
import torch
import random
import sys
import os
import time
import argparse
from scipy.stats import beta, norm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from typing import List, Any, Tuple
from rouge_score import rouge_scorer

# ==============================================================================
# 0. Argument Parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "DP-ICL 'G-baseline' (black-box style): single-prompt extraction attack baseline.\n"
            "Default configuration assumes SAMSum-like CSV with columns (id, dialogue, summary),\n"
            "and performs extraction on the SUMMARY column. You can use other datasets ONLY if\n"
            "you set the correct column indices / text field."
        )
    )

    # Paths
    p.add_argument("--local_model_path", type=str,
                   default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF",
                   help="Path to local HuggingFace causal LM.")
    p.add_argument("--data_path", type=str,
                   default="/local/scratch/yxia327/data/Samsum/train.csv",
                   help="CSV dataset path.")

    # CSV parsing
    p.add_argument("--data_encoding", type=str, default="utf-8",
                   help="Encoding for reading CSV.")
    p.add_argument("--errors", type=str, default="ignore",
                   help="Encoding error handling: 'ignore'/'replace'/etc.")
    p.add_argument("--delimiter", type=str, default=",",
                   help="CSV delimiter.")
    p.add_argument("--has_header", action="store_true", default=True,
                   help="Treat first row as header (default True). Use --no_header to disable.")
    p.add_argument("--no_header", action="store_false", dest="has_header",
                   help="Dataset has no header row.")

    # Dataset columns (default assumes: id, dialogue, summary)
    p.add_argument("--id_col", type=int, default=0, help="Column index for id.")
    p.add_argument("--dialogue_col", type=int, default=1, help="Column index for dialogue (unused in baseline).")
    p.add_argument("--summary_col", type=int, default=2, help="Column index for summary/text used in attack.")
    p.add_argument("--text_col", type=int, default=None,
                   help="Optional: if set, use this column as the attack text instead of --summary_col.")

    # Experiment constants / controls
    p.add_argument("--rouge_l_threshold", type=float, default=0.7,
                   help="ROUGE-L F1 threshold to count as a successful extraction.")
    p.add_argument("--prefix_words", type=int, default=5,
                   help="Number of leading words used as the canary prefix.")
    p.add_argument("--shots_per_subset", type=int, default=1,
                   help="Number of exemplars per subset (shots). Total exemplars = shots_per_subset * K. "
                        "Default 1 matches original code (num_exemplars = K).")

    # Audit trials
    p.add_argument("--T_audit_baseline", type=int, default=400,
                   help="Number of audit trials for the baseline extraction attack.")

    # Generation settings (default matches original: do_sample=True, temperature=0.9, max_new_tokens=50)
    p.add_argument("--max_new_tokens", type=int, default=50,
                   help="Max new tokens for generation.")
    p.add_argument("--do_sample", action="store_true", default=True,
                   help="Use sampling (default True). Use --no_sample to disable.")
    p.add_argument("--no_sample", action="store_false", dest="do_sample",
                   help="Disable sampling (greedy/beam depending on other params).")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature.")

    # GDP conversion delta (kept as a parameter for reproducibility)
    p.add_argument("--delta", type=float, default=1e-5,
                   help="Delta for GDP->(eps,delta) conversion (baseline uses a fixed delta in original code).")

    # Batch experiment settings
    p.add_argument("--subsets_list", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 14],
                   help="List of K values (number of subsets/ensembles).")
    p.add_argument("--num_runs", type=int, default=5, help="Number of runs per K.")
    p.add_argument("--output_dir", type=str, default="experiment_results_baseline",
                   help="Output directory for logs.")
    p.add_argument("--output_prefix", type=str, default="baseline_black",
                   help="Output filename prefix (default 'baseline_black').")

    # Seeding
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (0 = do not force seeding). If >0, reseeds per (run_id, K).")

    return p.parse_args()


def maybe_seed(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


# ==============================================================================
# 1. Privacy Audit Helper Functions
# ==============================================================================

def clopper_pearson(successes: int, trials: int, confidence_level: float) -> Tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)


def gdp_to_eps(mu: float, delta: float) -> float:
    if mu <= 0:
        return 0.0
    min_eps, max_eps = 0.0, 100.0
    for _ in range(100):
        eps = (min_eps + max_eps) / 2
        term1 = norm.cdf(-eps / mu + mu / 2)
        term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        if (term1 - term2) > delta:
            min_eps = eps
        else:
            max_eps = eps
    return float(min_eps)


# ==============================================================================
# 2. The Baseline Extraction Attack
# ==============================================================================

def run_extraction_attack(
    llm_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    exemplars_with: List[str],
    exemplars_without: List[str],
    canary: str,
    canary_prefix: str,
    T_audit: int,
    rouge_l_threshold: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    delta: float,
    f: Any,  # File handle for logging
) -> float:
    """
    Runs the single-prompt extraction attack and returns an empirical ε lower bound.
    """
    print(f"\n--- AUDIT: Running Single-Prompt Extraction Attack (Baseline) ---", file=f)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    prompt_template = """Here is a collection of text passages:

--- CONTEXT START ---
{context_string}
--- CONTEXT END ---

Based on the text provided above, please complete the following sentence:

"{prefix}"
"""
    success_with_canary = 0
    success_without_canary = 0

    # Part 1: with canary in context
    print("Running baseline attack with canary in context...")
    context_string_with = "\n".join(exemplars_with)
    prompt_with = prompt_template.format(context_string=context_string_with, prefix=canary_prefix)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        temperature=temperature,
    )

    for _ in tqdm(range(T_audit), desc="Attacking with canary"):
        inputs = tokenizer(prompt_with, return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**inputs, generation_config=gen_cfg)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        rouge_scores = scorer.score(canary, canary_prefix + " " + generated_text)
        if rouge_scores["rougeL"].fmeasure > rouge_l_threshold:
            success_with_canary += 1

    # Part 2: without canary in context
    print("Running baseline attack without canary in context...")
    context_string_without = "\n".join(exemplars_without)
    prompt_without = prompt_template.format(context_string=context_string_without, prefix=canary_prefix)

    for _ in tqdm(range(T_audit), desc="Attacking without canary"):
        inputs = tokenizer(prompt_without, return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**inputs, generation_config=gen_cfg)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        rouge_scores = scorer.score(canary, canary_prefix + " " + generated_text)
        if rouge_scores["rougeL"].fmeasure > rouge_l_threshold:
            success_without_canary += 1

    # Part 3: calculate result
    if T_audit == 0:
        return 0.0

    tpr = success_with_canary / T_audit
    fpr = success_without_canary / T_audit
    fp_count, fn_count = success_without_canary, T_audit - success_with_canary

    print(
        f"TPR = {tpr:.4f} ({success_with_canary}/{T_audit}) | "
        f"FPR = {fpr:.4f} ({success_without_canary}/{T_audit})",
        file=f,
    )

    _, alpha_upper = clopper_pearson(fp_count, T_audit, 0.95)
    _, beta_upper = clopper_pearson(fn_count, T_audit, 0.95)
    mu_lower_bound = max(0.0, float(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper)))
    eps_lower_bound = gdp_to_eps(mu_lower_bound, delta)

    print(f"Result (Extraction Attack): μ_lower={mu_lower_bound:.4f}, ε_lower={eps_lower_bound:.4f}", file=f)
    return float(eps_lower_bound)


# ==============================================================================
# 3. Main Experiment Procedure
# ==============================================================================

def run_baseline_experiment(
    output_filename: str,
    num_subsets: int,
    all_data: List[Tuple[str, ...]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args,
):
    with open(output_filename, "w", encoding="utf-8") as f:
        print("=" * 20 + f" Baseline Extraction Attack (K={num_subsets}) " + "=" * 20, file=f)

        if args.shots_per_subset < 1:
            print("Error: --shots_per_subset must be >= 1", file=f)
            return {}

        # Data preparation: sample base exemplars + 1 canary
        num_exemplars = args.shots_per_subset * num_subsets
        total_samples = num_exemplars + 1
        if total_samples > len(all_data):
            print(f"Warning: Not enough data for K={num_subsets}. Skipping.", file=f)
            return {}

        selected_indices = random.sample(range(len(all_data)), total_samples)
        canary_index, base_indices = selected_indices[-1], selected_indices[:-1]

        text_col = args.text_col if args.text_col is not None else args.summary_col

        canary_fact = all_data[canary_index][text_col]
        canary_words = str(canary_fact).split()
        prefix_n = min(args.prefix_words, len(canary_words)) if len(canary_words) > 0 else 0
        canary_prefix = " ".join(canary_words[:prefix_n])

        print(f"Selected Canary Text (col={text_col}): {canary_fact}", file=f)
        print(f"Canary Prefix for Attack (first {prefix_n} words): {canary_prefix}\n", file=f)

        exemplars_without = [all_data[i][text_col] for i in base_indices]
        exemplars_with = list(exemplars_without)
        exemplars_with[random.randint(0, len(exemplars_with) - 1)] = canary_fact

        eps_ext = run_extraction_attack(
            llm_model=model,
            tokenizer=tokenizer,
            exemplars_with=exemplars_with,
            exemplars_without=exemplars_without,
            canary=canary_fact,
            canary_prefix=canary_prefix,
            T_audit=args.T_audit_baseline,
            rouge_l_threshold=args.rouge_l_threshold,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            delta=args.delta,
            f=f,
        )

        results = {"extraction_attack_eps": eps_ext}
        print(f"\nAudit finished for K={num_subsets}. Results saved to '{output_filename}'")
        return results


# ==============================================================================
# 4. Main Execution
# ==============================================================================

def load_dataset(args) -> List[Tuple[str, ...]]:
    try:
        with open(args.data_path, "r", encoding=args.data_encoding, errors=args.errors) as f:
            rows = [tuple(r) for r in csv.reader(f, delimiter=args.delimiter)]
            if args.has_header and len(rows) > 0:
                rows = rows[1:]
            return rows
    except FileNotFoundError:
        print(f"Error: Data file not found at `{args.data_path}`.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        sys.exit(1)


def load_model(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)
    print("Loading LLM model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Setup complete.")
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()

    maybe_seed(args.seed)

    all_data = load_dataset(args)
    if len(all_data) == 0:
        print("Error: loaded dataset is empty.")
        sys.exit(1)

    # Sanity check column count
    max_idx = max(args.id_col, args.dialogue_col, args.summary_col, *( [args.text_col] if args.text_col is not None else [] ))
    if len(all_data[0]) <= max_idx:
        print(f"Error: dataset row has {len(all_data[0])} columns, but you requested col index {max_idx}.")
        sys.exit(1)

    model, tokenizer = load_model(args)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for num_subsets in args.subsets_list:
        for run_id in range(1, args.num_runs + 1):
            if args.seed and args.seed > 0:
                # Per-run deterministic reseed
                maybe_seed(args.seed + run_id + 1000 * num_subsets)

            key = (run_id, num_subsets)
            output_filename = os.path.join(
                args.output_dir,
                f"{args.output_prefix}-run{run_id}-subsets{num_subsets}.txt"
            )

            print(f"\nStarting Run {run_id}/{args.num_runs} for K={num_subsets} subsets...")
            all_results[key] = run_baseline_experiment(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_data=all_data,
                model=model,
                tokenizer=tokenizer,
                args=args,
            )

    print("\n\n" + "=" * 30 + "\nFinal Summary of All Baseline Runs\n" + "=" * 30)
    for key, res in all_results.items():
        run_id, num_subsets = key
        if not res:
            continue
        print(f"Run {run_id}, Subsets {num_subsets}:")
        print(f"  -> ε_ext={res['extraction_attack_eps']:.4f}")
