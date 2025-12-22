import argparse
import os
import random
import math
import sys
import csv
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import beta, norm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# ==============================================================================
# 0. Set Seed for Reproducibility
# ==============================================================================
def set_seed(seed: int):
    """
    Set all relevant random seeds to ensure experiment reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. Hugging Face Transformers Setup (loaded in main after parsing args)
# ==============================================================================
tokenizer = None
model = None


def load_hf_model(local_model_path: str):
    global tokenizer, model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print(f"Model loaded on device: {model.device}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


# ==============================================================================
# 2. LLM Classifier for Normal Classification (TREC)
# ==============================================================================
def llm_classifier(contexts: List[str], query_article: str, exemplars_labels: List[str]) -> str:
    """
    Use a local Hugging Face Transformers model to perform TREC question classification.

    NOTE: This script is designed specifically for the TREC dataset (coarse labels).
    Labels: DESC, ENTY, ABBR, HUM, NUM, LOC
    """
    if tokenizer is None or model is None:
        raise RuntimeError("Tokenizer/model not loaded. Call load_hf_model(...) before running.")

    formatted_exemplars = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_exemplars += f"Text: {text}\nLabel: {label}\n\n"

    prompt_content = f"""You are a highly accurate text classifier. Classify the following text into one of the provided categories based on the examples.
The possible labels are: DESC, ENTY, ABBR, HUM, NUM, LOC. Only output the label.

Examples:
{formatted_exemplars}

Query Text: {query_article}
Label:"""

    messages = [{"role": "user", "content": prompt_content}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )
    outputs = model.generate(input_ids, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    valid_labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
    for label in valid_labels:
        if response.startswith(label):
            return label
    return np.random.choice(valid_labels)  # fallback if parsing fails


# ==============================================================================
# 3. DP-ICL Implementation for Auditing
# ==============================================================================
class DPICLForClassification:
    def __init__(self, llm_callable, class_labels: List[str], epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sigma = (math.sqrt(2) * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon

    def _get_votes(self, exemplars: List[Tuple[str, str]], query_str: str, num_subsets: int) -> List[str]:
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        votes = []
        for subset_indices in partition_indices:
            if len(subset_indices) == 0:
                continue
            subset_exemplars = [exemplars[i][0] for i in subset_indices]
            subset_labels = [exemplars[i][1] for i in subset_indices]
            prediction = self.llm(
                contexts=subset_exemplars,
                query_article=query_str,
                exemplars_labels=subset_labels
            )
            votes.append(prediction)
        return votes

    def get_clean_vote_vector(self, exemplars: List[Tuple[str, str]], query_str: str, num_subsets: int) -> np.ndarray:
        votes = self._get_votes(exemplars, query_str, num_subsets)
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        return histogram

    def __call__(self, exemplars: List[Tuple[str, str]], query_str: str, num_subsets: int) -> Tuple[str, np.ndarray]:
        clean_histogram = self.get_clean_vote_vector(exemplars, query_str, num_subsets)
        noisy_histogram = clean_histogram + np.random.normal(0, self.sigma, self.num_classes)
        winner_index = np.argmax(noisy_histogram)
        return (self.id_to_label[winner_index], noisy_histogram)


# ==============================================================================
# 4. Privacy Audit Helper Functions
# ==============================================================================
def clopper_pearson(successes, trials, confidence_level):
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)


def gdp_to_eps(mu, delta):
    if mu == 0:
        return 0
    min_eps, max_eps = 0, 100
    for _ in range(100):
        eps = (min_eps + max_eps) / 2
        term1 = norm.cdf(-eps / mu + mu / 2)
        term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        delta_val = term1 - term2
        if delta_val > delta:
            min_eps = eps
        else:
            max_eps = eps
    return min_eps


def collect_vote_vectors(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors ---")
    vectors_without = [
        dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets)
        for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")
    ]
    vectors_with = [
        dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")
    ]
    return vectors_with, vectors_without


def simulate_histograms_with_bootstrap(collected_vectors, sigma, num_classes, T_simulation):
    if not collected_vectors:
        return np.array([])
    collected_vectors_arr = np.array(collected_vectors)
    num_collected = len(collected_vectors_arr)
    bootstrap_indices = np.random.randint(0, num_collected, size=T_simulation)
    bootstrapped_clean_histograms = collected_vectors_arr[bootstrap_indices]
    noise = np.random.normal(0, sigma, size=(T_simulation, num_classes))
    noisy_histograms = bootstrapped_clean_histograms + noise
    return noisy_histograms


def run_audit_with_fixed_threshold_and_fdp_calc(dp_classifier, histograms_with, histograms_without, canary_class_label, f):
    print(f"\n--- STAGE 3: Running analysis on {len(histograms_with)} simulated histograms ---", file=f)
    print("Method: Fixed threshold (Yes if canary_score >= max_other_score) with f-DP bound calculation.", file=f)

    canary_class_id = dp_classifier.label_to_id[canary_class_label]
    non_canary_class_ids = [i for i in range(dp_classifier.num_classes) if i != canary_class_id]
    T_audit = len(histograms_with)
    confidence = 0.95

    scores_canary_with = histograms_with[:, canary_class_id]
    scores_non_canary_with = np.max(histograms_with[:, non_canary_class_ids], axis=1)
    scores_canary_without = histograms_without[:, canary_class_id]
    scores_non_canary_without = np.max(histograms_without[:, non_canary_class_ids], axis=1)

    tp_count = np.sum(scores_canary_with >= scores_non_canary_with)
    fp_count = np.sum(scores_canary_without >= scores_non_canary_without)
    fn_count = T_audit - tp_count

    tpr = tp_count / T_audit if T_audit > 0 else 0
    fpr = fp_count / T_audit if T_audit > 0 else 0
    fnr = fn_count / T_audit if T_audit > 0 else 0

    print("\n--- Empirical Rates (Fixed Threshold) ---", file=f)
    print(f"True Positive Rate (TPR)  = {tpr:.4f} ({tp_count}/{T_audit})", file=f)
    print(f"False Positive Rate (FPR) = {fpr:.4f} ({fp_count}/{T_audit})", file=f)
    print(f"False Negative Rate (FNR) = {fnr:.4f} ({fn_count}/{T_audit})", file=f)

    _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
    _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
    print(f"\n{confidence*100}% CI upper bound for α (FPR): {alpha_upper:.4f}", file=f)
    print(f"{confidence*100}% CI upper bound for β (FNR): {beta_upper:.4f}", file=f)

    if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0:
        print("\nCould not compute a meaningful μ value due to invalid CI bounds (0 or 1).", file=f)
        return

    mu_lower_bound1 = norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper)
    mu_lower_bound2 = norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)
    mu_lower_bound = max(mu_lower_bound1, mu_lower_bound2)

    print("\n--- f-DP Bound Calculation ---", file=f)
    print(f"Empirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)


# ==============================================================================
# 5. Main procedure (TREC)
# ==============================================================================
def procedure(
    output_filename: str,
    num_subsets: int,
    all_data: List[Tuple[str, str]],
    llm_classifier_func,
    T_collect: int,
    T_simulation: int,
    collect_epsilon: float,
    epsilons: List[float],
    delta: float,
):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("=" * 20 + " DP-ICL Membership Inference Mislabeled Audit (TREC) " + "=" * 20, file=f)

        class_labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

        # --- Randomly Sample Canary and Base Exemplars ---
        num_base_exemplars = 2 * num_subsets
        if len(all_data) <= num_base_exemplars:
            raise ValueError("Not enough data to sample for the audit.")

        # Randomly select a canary from the TREC dataset
        canary_record = random.choice(all_data)
        canary_record_text = canary_record[0]

        # Dynamically determine its true and a mislabeled label
        true_label = canary_record[1]
        possible_mislabels = [label for label in class_labels if label != true_label]
        mislabeled_label = random.choice(possible_mislabels)

        # For the audit, the canary exemplar is paired with its TRUE label
        canary_exemplar_labeled = (canary_record_text, true_label)
        canary_class_label = true_label

        # Randomly select base exemplars from the rest of the data
        other_data = [rec for rec in all_data if rec != canary_record]
        base_exemplars = random.sample(other_data, num_base_exemplars)

        # Create the two adjacent datasets for the audit
        exemplars_with_labeled = [canary_exemplar_labeled] + base_exemplars[1:]
        exemplars_without_labeled = base_exemplars

        # NOTE: stage-1 uses a fixed epsilon for collecting clean vote vectors
        dp_classifier_collect = DPICLForClassification(
            llm_callable=llm_classifier_func,
            class_labels=class_labels,
            epsilon=collect_epsilon,
            delta=delta
        )

        print(f"Random Canary Text: '{canary_record_text[:100]}...'", file=f)
        print(f"Canary True Label: {true_label}, Mislabeled as: {mislabeled_label}", file=f)

        # Stage 1: Collect raw vote vectors
        collected_vectors_with, collected_vectors_without = collect_vote_vectors(
            dp_classifier=dp_classifier_collect,
            exemplars_with=exemplars_with_labeled,
            exemplars_without=exemplars_without_labeled,
            query=canary_record_text,  # raw text query
            num_subsets=num_subsets,
            T_collect=T_collect
        )

        mean_vector_with = np.mean(collected_vectors_with, axis=0) if collected_vectors_with else np.zeros(dp_classifier_collect.num_classes)
        mean_vector_without = np.mean(collected_vectors_without, axis=0) if collected_vectors_without else np.zeros(dp_classifier_collect.num_classes)
        print(f"\nNumber of Subsets (K): {num_subsets}", file=f)
        print(f"\nComputed Mean Vector (With Canary): {mean_vector_with}", file=f)
        print(f"Computed Mean Vector (Without Canary): {mean_vector_without}", file=f)

        # Stage 2+3: simulate & audit for each epsilon
        for epsilon in epsilons:
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier_func,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=delta
            )

            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms using BOOTSTRAP RESAMPLING ---")
            print(f"Theoretical ε={dp_classifier.epsilon}, δ={dp_classifier.delta}", file=f)

            histograms_with = simulate_histograms_with_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation
            )
            histograms_without = simulate_histograms_with_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation
            )

            # Stage 3: Run analysis
            run_audit_with_fixed_threshold_and_fdp_calc(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label,
                f=f
            )

    print(f"\nAudit finished. Results saved to '{output_filename}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DP-ICL mislabel audit script (specialized for the TREC dataset)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/local/scratch/yxia327/data/TREC/test.csv",
        help="Path to TREC test.csv (coarse labels). NOTE: This script assumes TREC format/mapping."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF",
        help="Local HuggingFace model path."
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10, 12, 14],
        help="List of K values (#subsets/partitions) to run."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of repeated runs per K."
    )
    parser.add_argument(
        "--T_collect",
        type=int,
        default=200,
        help="Number of clean vote vectors to collect (stage 1)."
    )
    parser.add_argument(
        "--T_simulation",
        type=int,
        default=400000,
        help="Number of bootstrap simulations per epsilon (stage 2)."
    )
    parser.add_argument(
        "--collect_epsilon",
        type=float,
        default=1.0,
        help="Epsilon used in stage 1 when collecting clean vote vectors (kept fixed by default)."
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
        help="List of epsilons to analyze in stage 2/3."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Delta used for GDP->(eps,delta) conversion and noise calibration."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results_mislabel",
        help="Directory to save output txt files."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="trec-run",
        help="Output filename prefix. Final name: {output_prefix}{run_id}-subsets{K}.txt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. If 0, no explicit seeding is applied."
    )
    return parser.parse_args()


def load_trec_data(data_path: str) -> List[Tuple[str, str]]:
    """
    Load TREC test.csv and return a list of (text, coarse_label_str).

    Expected mapping (coarse label id -> class):
      0: DESC, 1: ENTY, 2: ABBR, 3: HUM, 4: NUM, 5: LOC

    NOTE: This loader is specialized for TREC's CSV format used in your project:
    - It reads with pandas and uses row[0] as coarse label id and row[2] as text.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for loading TREC csv. Please install pandas.") from e

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: TREC data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    all_data_raw = df.iloc[1:].values.tolist()  # keep original behavior: drop header row

    coarse_to_label = {
        "0": "DESC",
        "1": "ENTY",
        "2": "ABBR",
        "3": "HUM",
        "4": "NUM",
        "5": "LOC"
    }

    # Basic sanity check: expect at least 3 columns, with text in col 2
    for i, row in enumerate(all_data_raw[:5]):
        if len(row) < 3:
            raise ValueError(
                f"TREC CSV format mismatch: expected >=3 columns, got {len(row)} at row index {i}."
            )

    all_data = [(row[2], coarse_to_label.get(str(row[0]), "DESC")) for row in all_data_raw]
    return all_data


# ==============================================================================
# Entry
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()

    print("NOTE: This script is specialized for the TREC dataset (coarse labels). "
          "If you use another dataset/format, label mapping and columns will be wrong.")

    if args.seed != 0:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data (TREC)
    all_data = load_trec_data(args.data_path)

    # Load local model
    load_hf_model(args.model_path)

    # Run experiments
    for num_subsets in args.num_subsets:
        for run_id in range(1, args.num_runs + 1):
            output_filename = os.path.join(
                args.output_dir,
                f"{args.output_prefix}-{run_id}-subsets{num_subsets}.txt"
            )
            procedure(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_data=all_data,
                llm_classifier_func=llm_classifier,
                T_collect=args.T_collect,
                T_simulation=args.T_simulation,
                collect_epsilon=args.collect_epsilon,
                epsilons=args.epsilons,
                delta=args.delta,
            )
