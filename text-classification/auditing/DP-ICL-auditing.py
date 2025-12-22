import numpy as np
import math
import csv
from scipy.stats import beta, norm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import random
import argparse
import os

# ==============================================================================
# 1. Hugging Face Transformers Setup (本地模型配置)
# ... (此部分代码无任何改动) ...
# ==============================================================================
# Specify the path to your local model
# The model/tokenizer are loaded in main() so we can parameterize the model path.
tokenizer = None
model = None

def load_hf_model(model_path: str):
    """Load a local Hugging Face causal LM and tokenizer."""
    global tokenizer, model
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
# ==============================================================================
# 2. LLM Classifier for Yes/No Task
# ... (此部分代码无任何改动) ...
# ==============================================================================
def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    使用本地 Hugging Face Transformers 模型进行成员推断。
    此函数直接询问 query_article 是否出现在上下文中。
    (已更新以包含 attention_mask)
    """
    formatted_context = ""
    for text in contexts:
        formatted_context += f"- {text}\n"
    
#     prompt_content = f"""You are a membership inference classifier.
# Below are some example texts:

# {formatted_context}

# Here is a query text:
# {query_article}

# Compare the query text to the context. Answer strictly 'Yes' if the query text appears inany of the examples, or No.​"""

    prompt_content = f"""You are a highly accurate text verification system. Your sole function is to verify if the text in the <query> tag is present in the <context> tag.
Your response must be a single word: either 'Yes' or 'No'.

<context>
{formatted_context}
</context>

<query>
{query_article}
</query>

<response>
"""
    messages = [{"role": "user", "content": prompt_content}]

    # 变化 1: 先格式化 prompt，再用 tokenizer 获取包含 attention_mask 的字典
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(max_new_tokens=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    
    # 变化 2: 使用 **inputs 将 input_ids 和 attention_mask 一同传入
    outputs = model.generate(**inputs, generation_config=generation_config)

    # 变化 3: 在解码时使用正确的输入长度 (inputs['input_ids'])
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip().lower()
    
    valid_labels = ["Yes", "No"]
    if ("yes" in response) or ("Yes" in response):
        return "Yes"
    elif ("no" in response) or ("No" in response):
        return "No"
    return np.random.choice(valid_labels)

# ==============================================================================
# 3. DP-ICL Implementation - Refactored for Auditing
# ... (此部分代码无任何改动) ...
# ==============================================================================
class DPICLForClassification:
    # This class is generic and requires no changes. It will work with the new labels.
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = (math.sqrt(2) * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon

    def _get_votes(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> list[str]:
        """Private method to handle partitioning and LLM calls, returning votes."""
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        votes = []
        for subset_indices in partition_indices:
            if len(subset_indices) == 0: continue
            subset_exemplars = [exemplars[i][0] for i in subset_indices]
            subset_labels = [exemplars[i][1] for i in subset_indices]
            prediction = self.llm(contexts=subset_exemplars, query_article=query_str, exemplars_labels=subset_labels)
            votes.append(prediction)
        return votes

    def get_clean_vote_vector(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> np.ndarray:
        """Runs the mechanism and returns the pre-noise vote histogram."""
        votes = self._get_votes(exemplars, query_str, num_subsets)
        histogram = np.zeros(self.num_classes)
        # print(votes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        return histogram

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> tuple[str, np.ndarray]:
        """Runs the full DP-ICL process, returning the noisy result."""
        clean_histogram = self.get_clean_vote_vector(exemplars, query_str, num_subsets)
        noisy_histogram = clean_histogram + np.random.normal(0, self.sigma, self.num_classes)
        winner_index = np.argmax(noisy_histogram)
        return (self.id_to_label[winner_index], noisy_histogram)

def clopper_pearson(successes, trials, confidence_level):
    if trials == 0: return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)

def gdp_to_eps(mu, delta):
    if mu == 0: return 0
    min_eps, max_eps = 0, 100
    for _ in range(100):
        eps = (min_eps + max_eps) / 2
        term1 = norm.cdf(-eps / mu + mu / 2)
        term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        delta_val = term1 - term2
        if delta_val > delta: min_eps = eps
        else: max_eps = eps
    return min_eps

def collect_vote_vectors_for_resampling(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect,f):
    """
    STAGE 1: Collect a set of clean vote vectors to build an empirical distribution. (Expensive)
    """
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors to build empirical distribution ---")
    
    vectors_without = []
    average_vector_without = np.zeros(dp_classifier.num_classes)
    for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets)
        vectors_without.append(clean_vector)
        average_vector_without += clean_vector
    average_vector_without /= T_collect
    print(f"Computed Mean Vector (Without Canary): {average_vector_without}",file = f)


    vectors_with = []
    average_vector_with = np.zeros(dp_classifier.num_classes)
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        vectors_with.append(clean_vector)
        average_vector_with += clean_vector
    average_vector_with /= T_collect
    print(f"Computed Mean Vector (With Canary): {average_vector_with}",file = f)
    return vectors_with, vectors_without

def simulate_histograms_from_bootstrap(collected_vectors, sigma, num_classes, T_simulation):
    """
    STAGE 2: Simulate noisy histograms by resampling from the empirical distribution (Bootstrap). (Cheap)
    """
    histograms = []
    for _ in range(T_simulation):
        # 1. Resample a clean vector with replacement from the collected empirical distribution.
        clean_vector = random.choice(collected_vectors)
        
        # 2. Add fresh Gaussian noise to the resampled vector.
        noisy_hist = clean_vector + np.random.normal(0, sigma, num_classes)
        histograms.append(noisy_hist)
    return histograms

def run_fdp_audit_on_histograms(dp_classifier, histograms_with, histograms_without, canary_class_label, f):
    """
    STAGE 3: Run f-DP statistical analysis on the simulated histograms. (Fast & Unchanged)
    """
    # This function remains the same as it operates on the final histograms.
    print(f"\n--- STAGE 3: Running f-DP analysis on {len(histograms_with)} simulated histograms ---", file=f)
    canary_class_id = dp_classifier.label_to_id[canary_class_label]
    T_audit = len(histograms_with)
    observations_with_target = []
    for hist in histograms_with:
        mx = -float('inf')
        for j in range(len(hist)):
            if j != canary_class_id: mx = max(mx, hist[j])
        observations_with_target.append(hist[canary_class_id] - mx)
    observations_without_target = []
    for hist in histograms_without:
        mx = -float('inf')
        for j in range(len(hist)):
            if j != canary_class_id: mx = max(mx, hist[j])
        observations_without_target.append(hist[canary_class_id] - mx)
    obs_with_np = np.array(observations_with_target)
    obs_without_np = np.array(observations_without_target)
    obs_with_np.sort(); obs_without_np.sort()
    candidate_thresholds = np.unique(np.concatenate((obs_with_np, obs_without_np)))
    max_eps = -float('inf')
    optimal_threshold, false_positives, false_negatives = None, -1, -1
    confidence = 0.95
    
    single_max_eps = -float('inf')
    threshold_for_single = 0
    fp_count=T_audit-np.searchsorted(obs_without_np, 0, side='right')
    fn_count=np.searchsorted(obs_with_np, 0, side='right')
    _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
    _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
    if alpha_upper < 1 and beta_upper < 1 and alpha_upper > 0 and beta_upper > 0:
        single_max_eps =  max(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper),norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper))
        threshold_for_single = 0
    print(f"\nSingle threshold at 0 (single) gives μ: {single_max_eps:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(single_max_eps, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)

    print("Finding optimal decision threshold...")
    for threshold in tqdm(candidate_thresholds, desc="Finding Threshold"):
        fp_count = T_audit - np.searchsorted(obs_without_np, threshold, side='right')
        fn_count = np.searchsorted(obs_with_np, threshold, side='right')
        _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
        _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
        if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0: continue
        current_eps = max(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper),norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper))
        if current_eps > max_eps:
            max_eps, optimal_threshold, false_positives, false_negatives = current_eps, threshold, fp_count, fn_count
    fpr, fnr = false_positives / T_audit, false_negatives / T_audit
    print("\n--- Audit Analysis Results ---", file=f)
    print(f"Optimal Decision Threshold (Advantage Score) found: {optimal_threshold:.4f}", file=f)
    print(f"Corresponding μ: {max_eps:.4f}", file=f)
    print(f"  False Positive Rate (α) = {fpr:.4f} ({false_positives}/{T_audit})", file=f)
    print(f"  False Negative Rate (β) = {fnr:.4f} ({false_negatives}/{T_audit})", file=f)
    _, alpha_upper = clopper_pearson(false_positives, T_audit, confidence)
    _, beta_upper = clopper_pearson(false_negatives, T_audit, confidence)
    print(f"\n{confidence*100}% CI upper bound for α: {alpha_upper:.4f}", file=f)
    print(f"{confidence*100}% CI upper bound for β: {beta_upper:.4f}", file=f)
    if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0:
        print("\nCould not compute a meaningful μ value.", file=f)
        return
    mu_lower_bound = norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)
    print(f"\nEmpirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)

# ==============================================================================
# 6. <<< MODIFIED: Procedure function with randomized data preparation >>>
# ==============================================================================
def procedure(
    output_filename: str,
    num_subsets: int,
    all_texts: list[str],
    llm_classifier,
    *,
    shots_per_subset: int = 2,
    T_collect: int,
    T_simulation: int,
    epsilons: list[float],
    delta: float,
    collect_epsilon: float = 1.0,
    canary_mode: str = "fixed",
    canary_index: int = 1,
    seed: int | None = None,
):
    """
    Run the 3-stage (collect -> bootstrap -> f-DP) audit for a binary Yes/No membership task.

    This script is dataset-agnostic: it only consumes raw text strings from `all_texts`.
    """
    rng = random.Random(seed) if seed is not None else random

    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " Membership Inference f-DP Privacy Audit " + "="*20, file=f)

        class_labels = ["Yes", "No"]

        # --- Randomly sample data for the audit ---
        if shots_per_subset < 1:
            raise ValueError(f"--shots_per_subset must be >= 1 (got {shots_per_subset}).")

        num_base_exemplars = int(shots_per_subset) * int(num_subsets)
        if len(all_texts) <= num_base_exemplars:
            raise ValueError(
                f"Not enough data to sample a canary and {num_base_exemplars} base exemplars "
                f"(got {len(all_texts)} texts)."
            )

        # 1) Choose a canary text
        if canary_mode == "fixed":
            idx = max(0, min(canary_index, len(all_texts) - 1))
        else:
            idx = rng.randrange(len(all_texts))
        canary_text = all_texts[idx]

        # 2) Choose base exemplars from the rest (labels for audit task are independent of dataset labels)
        remaining_indices = [i for i in range(len(all_texts)) if i != idx]
        base_indices = rng.sample(remaining_indices, num_base_exemplars)
        base_exemplars = [(all_texts[i], "No") for i in base_indices]

        # Adjacent datasets
        canary_class_label = "Yes"
        exemplars_without_target = base_exemplars
        exemplars_with_target = [(canary_text, "Yes")] + base_exemplars[1:]

        print(f"Selected canary for audit: '{canary_text[:100]}...'", file=f)

        # --- Stage 1: collect clean vote vectors (expensive) ---
        temp_dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=collect_epsilon,
            delta=delta,
        )
        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=temp_dp_classifier,
            exemplars_with=exemplars_with_target,
            exemplars_without=exemplars_without_target,
            query=canary_text,
            num_subsets=num_subsets,
            T_collect=T_collect,
            f=f,
        )

        print(f"\nNumber of Subsets (K): {num_subsets}", file=f)
        print(f"Collected {len(collected_vectors_with)} vectors for 'With Canary' distribution.", file=f)
        print(f"Collected {len(collected_vectors_without)} vectors for 'Without Canary' distribution.", file=f)

        # --- Stages 2 & 3: bootstrap simulate and analyze per epsilon ---
        for epsilon in epsilons:
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier,
                class_labels=class_labels,
                epsilon=float(epsilon),
                delta=delta,
            )
            print(f"\n--- Analyzing for Theoretical ε={dp_classifier.epsilon} ---")
            f.write(f"\n{'='*15} ANALYSIS for ε={epsilon} {'='*15}\n")

            histograms_with = simulate_histograms_from_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation
            )
            histograms_without = simulate_histograms_from_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation
            )

            run_fdp_audit_on_histograms(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label,
                f=f,
            )

    print(f"\nMembership inference audit finished. Results saved to '{output_filename}'.")


# ==============================================================================
# 7. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DP-ICL white-box audit (binary Yes/No membership). Dataset-agnostic: uses only raw text content."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF",
        help="Path to the local HuggingFace model checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/local/scratch/yxia327/data/ag_news_csv/test.csv",
        help="Path to a CSV file. Only the text column is used; dataset labels are ignored.",
    )
    parser.add_argument(
        "--text_col",
        type=int,
        default=2,
        help="0-based column index of the text field in the CSV (default: 2).",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Set this flag if the CSV has NO header row. By default, the script skips the first row (to match the original script).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter (default: ',').",
    )

    # Experiment parameters
    parser.add_argument(
        "--num_subsets",
        type=int,
        nargs="+",
        default=[4],
        help="K values (#subsets/partitions). Accepts one or more ints. Default matches original script: 4.",
    )

    parser.add_argument(
        "--shots_per_subset",
        type=int,
        default=2,
        help="Number of base exemplars per subset (shots per subset). Total base exemplars = shots_per_subset * K. Default 2 matches original num_base_exemplars=2*K.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs per K (default matches original script: 1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results_white",
        help="Directory to write result txt files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="sample-run",
        help="Prefix for output files. Output is '{prefix}{run_id}-subsets{K}.txt'.",
    )

    parser.add_argument("--T_collect", type=int, default=200, help="Stage 1: number of clean vectors to collect.")
    parser.add_argument("--T_simulation", type=int, default=400000, help="Stage 2: bootstrap simulations.")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1.0],
        help="Theoretical epsilons to analyze (default matches original script: [1.0]).",
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for DP/GDP conversion.")
    parser.add_argument(
        "--collect_epsilon",
        type=float,
        default=1.0,
        help="Epsilon used when instantiating the temporary classifier for Stage 1 (does not affect clean votes).",
    )

    # Canary selection (to keep backward-compatibility, default is fixed index=1)
    parser.add_argument(
        "--canary_mode",
        choices=["fixed", "random"],
        default="fixed",
        help="How to choose canary text: fixed index or random (default matches original: fixed).",
    )
    parser.add_argument(
        "--canary_index",
        type=int,
        default=1,
        help="When --canary_mode fixed, use this 0-based index (default matches original: 1).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. 0 means do not force determinism. If non-zero, run seed = seed + run_id.",
    )

    args = parser.parse_args()

    print("NOTE: This script is dataset-agnostic: it ignores dataset labels and only uses raw text content.")
    print(f"Reading texts from CSV: {args.data_path} (text_col={args.text_col}, has_header={not args.no_header})")

    # Load model/tokenizer
    load_hf_model(args.model_path)

    # Load texts
    all_texts: list[str] = []
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=args.delimiter)
            if not args.no_header:
                next(reader, None)
            for row in reader:
                if row is None or len(row) <= args.text_col:
                    continue
                txt = str(row[args.text_col]).strip()
                if txt:
                    all_texts.append(txt)
    except FileNotFoundError:
        print(f"Error: data file not found: {args.data_path}")
        sys.exit(1)

    if len(all_texts) == 0:
        print("Error: no texts were loaded. Check --data_path/--text_col/--delimiter/--has_header.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for k in args.num_subsets:
        for run_id in range(args.num_runs):
            run_seed = None if args.seed == 0 else (args.seed + run_id)
            if run_seed is not None:
                random.seed(run_seed)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)

            output_filename = os.path.join(args.output_dir, f"{args.output_prefix}{run_id}-subsets{k}.txt")
            procedure(
                output_filename=output_filename,
                num_subsets=int(k),
                all_texts=all_texts,
                llm_classifier=llm_classifier,
                shots_per_subset=int(args.shots_per_subset),
                T_collect=int(args.T_collect),
                T_simulation=int(args.T_simulation),
                epsilons=[float(e) for e in args.epsilons],
                delta=float(args.delta),
                collect_epsilon=float(args.collect_epsilon),
                canary_mode=args.canary_mode,
                canary_index=int(args.canary_index),
                seed=run_seed,
            )
