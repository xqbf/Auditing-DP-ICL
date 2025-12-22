import numpy as np
import math
import csv
from scipy.stats import beta, norm
import torch
import random
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ==============================================================================
# NOTE
# ==============================================================================
# This script is specialized for the AG News dataset.
# It assumes the label mapping: 1/2/3/4 -> World/Sports/Business/Sci/Tech.
# If you use other datasets or other label conventions, the evaluation will be incorrect.
# ==============================================================================

# Globals set by init_model()
tokenizer = None
model = None

# ==============================================================================
# 0. Set Seed for Reproducibility
# ==============================================================================
def set_seed(seed: int):
    """Set all relevant random seeds to ensure experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. Model Loading
# ==============================================================================
def init_model(model_path: str):
    """Load tokenizer/model once and set global handles used by llm_classifier()."""
    global tokenizer, model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print(f"Model loaded on device: {model.device}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

# ==============================================================================
# 2. LLM Classifier for Normal Classification (AG News)
# ==============================================================================
def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    Use local Hugging Face Transformers model for standard classification (AG News).
    Only output the label.
    """
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer not initialized. Call init_model() first.")

    formatted_exemplars = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_exemplars += f"Text: {text}\nLabel: {label}\n\n"

    prompt_content = f"""You are a highly accurate text classifier. Classify the following text into one of the provided categories based on the examples.
The possible labels are: World, Sports, Business, Sci/Tech. Only output the label.

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
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()

    valid_labels = ["World", "Sports", "Business", "Sci/Tech"]
    for label in valid_labels:
        if response.startswith(label):
            return label

    # Fallback: random label if parsing fails
    return np.random.choice(valid_labels)

# ==============================================================================
# 3. DP-ICL Implementation (Classification)
# ==============================================================================
class DPICLForClassification:
    def __init__(self, llm_callable, class_labels: list[str], epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sigma = (math.sqrt(2) * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon

    def _get_votes(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> list[str]:
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)

        votes = []
        for subset_indices in partition_indices:
            if len(subset_indices) == 0:
                continue
            subset_exemplars = [exemplars[i][0] for i in subset_indices]
            subset_labels = [exemplars[i][1] for i in subset_indices]
            prediction = self.llm(
                contexts=subset_exemplars, query_article=query_str, exemplars_labels=subset_labels
            )
            votes.append(prediction)
        return votes

    def get_clean_vote_vector(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> np.ndarray:
        votes = self._get_votes(exemplars, query_str, num_subsets)
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        return histogram

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> tuple[str, np.ndarray]:
        clean_histogram = self.get_clean_vote_vector(exemplars, query_str, num_subsets)
        noisy_histogram = clean_histogram + np.random.normal(0, self.sigma, self.num_classes)
        winner_index = int(np.argmax(noisy_histogram))
        return (self.id_to_label[winner_index], noisy_histogram)

# ==============================================================================
# 4. Utility Test
# ==============================================================================
def _run_utility_test(dp_classifier: DPICLForClassification,
                      exemplars: list[tuple[str, str]],
                      test_set: list[tuple[str, str, str]],
                      num_subsets: int,
                      class_labels: list[str],
                      desc: str) -> tuple[int, int]:
    correct_count = 0
    total_count = 0
    for rec in tqdm(test_set, desc=desc):
        query_article = rec[2]
        true_label = class_labels[int(rec[0]) - 1]  # AG News: label in {1,2,3,4}

        predicted_label, _ = dp_classifier(
            exemplars=exemplars,
            query_str=query_article,
            num_subsets=num_subsets
        )

        if predicted_label == true_label:
            correct_count += 1
        total_count += 1

    return correct_count, total_count


def utility_test_procedure(output_filename: str,
                           num_subsets: int,
                           all_data: list[tuple[str, str, str]],
                           llm_classifier_func,
                           epsilons: list[float],
                           delta: float,
                           exemplar_start: int,
                           exemplar_end: int,
                           test_start: int,
                           test_end: int,
                           canary_text: str,
                           mislabeled_label: str):
    """
    Run utility test (accuracy drop) caused by adding a mislabeled exemplar into ICL context.
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("=" * 20 + " DP-ICL Classification Utility Test " + "=" * 20, file=f)
        class_labels = ["World", "Sports", "Business", "Sci/Tech"]

        # --- 1) Build base exemplars ---
        # (all_data already excludes header row; indices are 0-based)
        base_exemplars_data = all_data[exemplar_start:exemplar_end]
        base_exemplars = [(rec[2], class_labels[int(rec[0]) - 1]) for rec in base_exemplars_data]

        # --- 2) Build poisoned exemplar (mislabeled) ---
        canary_exemplar_mislabeled = (canary_text, mislabeled_label)

        exemplars_with_mislabeled = [canary_exemplar_mislabeled] + base_exemplars
        exemplars_without_mislabeled = base_exemplars

        # --- 3) Select test set ---
        test_set = all_data[test_start:test_end]
        T_samples = len(test_set)

        print(f"Number of Subsets (K): {num_subsets}", file=f)
        print(f"Test Samples (T): {T_samples} (from index {test_start} to {test_end})", file=f)
        print(f"Exemplar range: [{exemplar_start}, {exemplar_end})", file=f)
        print(f"Mislabeled exemplar label: {mislabeled_label}", file=f)

        # --- 4) Loop epsilons ---
        for epsilon in epsilons:
            print(f"\n--- Testing Epsilon (ε) = {epsilon} ---", file=f)
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier_func,
                class_labels=class_labels,
                epsilon=float(epsilon),
                delta=float(delta)
            )

            acc_with, total_with = _run_utility_test(
                dp_classifier, exemplars_with_mislabeled, test_set, num_subsets, class_labels,
                desc=f"Utility test (with mislabeled), ε={epsilon}, K={num_subsets}"
            )
            acc_without, total_without = _run_utility_test(
                dp_classifier, exemplars_without_mislabeled, test_set, num_subsets, class_labels,
                desc=f"Utility test (without mislabeled), ε={epsilon}, K={num_subsets}"
            )

            print(f"Accuracy (With Mislabeled) = {acc_with/total_with:.4f} ({acc_with}/{total_with})", file=f)
            print(f"Accuracy (Without Mislabeled) = {acc_without/total_without:.4f} ({acc_without}/{total_without})", file=f)

            utility_drop = (acc_without/total_without) - (acc_with/total_with)
            print(f"Utility Drop = {utility_drop:.4f}", file=f)

    print(f"\nUtility Test finished. Results saved to '{output_filename}'")


def parse_args():
    p = argparse.ArgumentParser(description="DP-ICL single-run utility test (AG News specialized).")

    # Data/model
    p.add_argument("--data_path", type=str,
                   default="/local/scratch/yxia327/data/ag_news_csv/test.csv",
                   help="Path to AG News CSV (same format as ag_news_csv/test.csv).")
    p.add_argument("--model_path", type=str,
                   default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF",
                   help="Local HuggingFace model path.")

    # Experiment knobs
    p.add_argument("--num_subsets", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 14],
                   help="One or more K values (number of partitions/subsets).")
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 8.0],
                   help="One or more epsilon values to test.")
    p.add_argument("--delta", type=float, default=1e-5, help="DP delta used to compute sigma.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed. 0 means do not force a seed (keeps original behavior).")

    # Indices / sizes (keep defaults identical to original code)
    p.add_argument("--exemplar_start", type=int, default=1,
                   help="Start index (0-based) in all_data for base exemplars (inclusive).")
    p.add_argument("--exemplar_end", type=int, default=31,
                   help="End index (0-based) in all_data for base exemplars (exclusive).")
    p.add_argument("--test_start", type=int, default=500,
                   help="Start index (0-based) for test set (inclusive).")
    p.add_argument("--test_end", type=int, default=1000,
                   help="End index (0-based) for test set (exclusive).")

    # Canary / mislabel
    p.add_argument("--canary_text", type=str,
                   default="Real Madrid has signed a new player.",
                   help="Text for the injected mislabeled exemplar.")
    p.add_argument("--mislabeled_label", type=str, default="Business",
                   choices=["World", "Sports", "Business", "Sci/Tech"],
                   help="Label to (incorrectly) assign to the injected exemplar.")

    # Output
    p.add_argument("--output_dir", type=str, default="experiment_results_single",
                   help="Directory to save results.")
    p.add_argument("--output_prefix", type=str, default="utility-test",
                   help="Prefix of output filenames. Final name: {prefix}-subsets{K}.txt")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("NOTE: This script is specialized for the AG News dataset.")
    print("      It assumes label mapping 1/2/3/4 -> World/Sports/Business/Sci/Tech.")
    print("      If you use other datasets, you must change label parsing and prompts accordingly.\n")

    if args.seed != 0:
        set_seed(args.seed)

    init_model(args.model_path)

    # Load AG News test CSV
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            # CSV header exists; keep original behavior: skip header row.
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        raise SystemExit(f"Error: data file not found: {args.data_path}")

    # Basic sanity checks (lightweight, non-fatal)
    if len(all_data) < max(args.exemplar_end, args.test_end) + 1:
        print(f"WARNING: all_data has only {len(all_data)} rows; "
              f"but exemplar_end/test_end request up to {max(args.exemplar_end, args.test_end)}.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for k in args.num_subsets:
        output_filename = os.path.join(args.output_dir, f"{args.output_prefix}-subsets{k}.txt")
        utility_test_procedure(
            output_filename=output_filename,
            num_subsets=k,
            all_data=all_data,
            llm_classifier_func=llm_classifier,
            epsilons=args.epsilons,
            delta=args.delta,
            exemplar_start=args.exemplar_start,
            exemplar_end=args.exemplar_end,
            test_start=args.test_start,
            test_end=args.test_end,
            canary_text=args.canary_text,
            mislabeled_label=args.mislabeled_label,
        )
