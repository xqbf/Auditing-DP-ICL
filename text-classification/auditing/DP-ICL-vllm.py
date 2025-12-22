import argparse
import csv
import math
import os
import random
import sys
from typing import List, Sequence, Tuple

import numpy as np
from openai import OpenAI
from scipy.stats import beta, norm
from tqdm import tqdm


# ==============================================================================
# 0. Utilities
# ==============================================================================

def _maybe_set_seed(seed: int) -> None:
    if seed and seed != 0:
        random.seed(seed)
        np.random.seed(seed)


def load_texts(
    data_path: str,
    data_format: str = "csv",
    text_col: int = 2,
    delimiter: str = ",",
    has_header: bool = True,
    encoding: str = "utf-8",
    max_rows: int | None = None,
) -> List[str]:
    """
    Dataset-agnostic loader: returns a list[str] of texts only.
    - csv: reads `text_col` column (0-indexed). If has_header, skips first row.
    - txt: one sample per line.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    texts: List[str] = []
    if data_format.lower() == "txt":
        with open(data_path, "r", encoding=encoding) as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    if data_format.lower() != "csv":
        raise ValueError(f"Unsupported --data_format: {data_format}. Use csv or txt.")

    with open(data_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if has_header and i == 0:
                continue
            if max_rows is not None and len(texts) >= max_rows:
                break
            if not row or text_col >= len(row):
                continue
            txt = row[text_col]
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt:
                texts.append(txt)
    return texts


# ==============================================================================
# 1. LLM classifier via vLLM OpenAI-compatible endpoint
# ==============================================================================

def make_llm_classifier(client: OpenAI, model_id: str, system_prompt: str, max_tokens: int, temperature: float):
    """
    Returns a callable: (contexts, query_article, exemplars_labels) -> "Yes"|"No"
    """
    def llm_classifier(contexts: List[str], query_article: str, exemplars_labels: List[str]) -> str:
        formatted_context = ""
        for text in contexts:
            formatted_context += f"- {text}\n"

        prompt_content = f"""You are a highly accurate text verification system.
Your sole function is to verify if the text in the <query> tag is present in the <context> tag.
Your response must be a single word: either 'Yes' or 'No'.

<context>
{formatted_context}
</context>

<query>
{query_article}
</query>

/no_think
"""

        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response_message = resp.choices[0].message
        output = None

        if response_message and response_message.content:
            response = response_message.content.strip()
            if response.lower().startswith("yes"):
                output = "Yes"
            elif response.lower().startswith("no"):
                output = "No"
            else:
                print(f"!!! Warning: unexpected model output: '{response}'. Falling back to random label.")
                output = random.choice(exemplars_labels)
        else:
            print("!!! Warning: model returned empty content. Falling back to random label.")
            output = random.choice(exemplars_labels)

        return output

    return llm_classifier


# ==============================================================================
# 2. DP-ICL mechanism (unchanged)
# ==============================================================================

class DPICLForClassification:
    def __init__(self, llm_callable, class_labels: List[str], epsilon: float, delta: float):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
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
            prediction = self.llm(contexts=subset_exemplars, query_article=query_str, exemplars_labels=subset_labels)
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
        winner_index = int(np.argmax(noisy_histogram))
        return (self.id_to_label[winner_index], noisy_histogram)


# ==============================================================================
# 3. Audit helper functions (unchanged)
# ==============================================================================

def clopper_pearson(successes: int, trials: int, confidence_level: float):
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)

def gdp_to_eps(mu: float, delta: float) -> float:
    if mu == 0:
        return 0.0
    min_eps, max_eps = 0.0, 100.0
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


# ==============================================================================
# 4. Three-stage bootstrap workflow (parameterized)
# ==============================================================================

def collect_vote_vectors_for_resampling(
    dp_classifier: DPICLForClassification,
    exemplars_with: List[Tuple[str, str]],
    exemplars_without: List[Tuple[str, str]],
    query: str,
    num_subsets: int,
    T_collect: int,
    f
):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors to build empirical distribution ---")

    vectors_without = []
    average_vector_without = np.zeros(dp_classifier.num_classes)
    for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets)
        vectors_without.append(clean_vector)
        average_vector_without += clean_vector
    average_vector_without /= T_collect
    print(f"Computed Mean Vector (Without Canary): {average_vector_without}", file=f)

    vectors_with = []
    average_vector_with = np.zeros(dp_classifier.num_classes)
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        vectors_with.append(clean_vector)
        average_vector_with += clean_vector
    average_vector_with /= T_collect
    print(f"Computed Mean Vector (With Canary): {average_vector_with}", file=f)

    return vectors_with, vectors_without


def simulate_histograms_from_bootstrap(collected_vectors, sigma: float, num_classes: int, T_simulation: int):
    histograms = []
    for _ in range(T_simulation):
        clean_vector = random.choice(collected_vectors)
        noisy_hist = clean_vector + np.random.normal(0, sigma, num_classes)
        histograms.append(noisy_hist)
    return histograms


def run_fdp_audit_on_histograms(
    dp_classifier: DPICLForClassification,
    histograms_with,
    histograms_without,
    canary_class_label: str,
    f,
    confidence: float = 0.95,
):
    print(f"\n--- STAGE 3: Running f-DP analysis on {len(histograms_with)} simulated histograms ---", file=f)
    canary_class_id = dp_classifier.label_to_id[canary_class_label]
    T_audit = len(histograms_with)

    observations_with_target = []
    for hist in histograms_with:
        mx = max(hist[j] for j in range(len(hist)) if j != canary_class_id)
        observations_with_target.append(hist[canary_class_id] - mx)

    observations_without_target = []
    for hist in histograms_without:
        mx = max(hist[j] for j in range(len(hist)) if j != canary_class_id)
        observations_without_target.append(hist[canary_class_id] - mx)

    obs_with_np = np.array(observations_with_target)
    obs_without_np = np.array(observations_without_target)
    obs_with_np.sort()
    obs_without_np.sort()

    candidate_thresholds = np.unique(np.concatenate((obs_with_np, obs_without_np)))

    max_eps = -float("inf")
    optimal_threshold, false_positives, false_negatives = None, -1, -1

    # Single threshold at 0 (for logging)
    fp_count_at_zero = T_audit - np.searchsorted(obs_without_np, 0, side="right")
    fn_count_at_zero = np.searchsorted(obs_with_np, 0, side="right")
    _, alpha_upper_at_zero = clopper_pearson(fp_count_at_zero, T_audit, confidence)
    _, beta_upper_at_zero = clopper_pearson(fn_count_at_zero, T_audit, confidence)
    single_mu = -float("inf")
    if 0 < alpha_upper_at_zero < 1 and 0 < beta_upper_at_zero < 1:
        single_mu = max(
            norm.ppf(1 - beta_upper_at_zero) - norm.ppf(alpha_upper_at_zero),
            norm.ppf(1 - alpha_upper_at_zero) - norm.ppf(beta_upper_at_zero),
        )
    print(f"\nSingle threshold at 0 gives μ: {single_mu:.4f}", file=f)

    print("Finding optimal decision threshold...")
    for threshold in tqdm(candidate_thresholds, desc="Finding Threshold"):
        fp_count = T_audit - np.searchsorted(obs_without_np, threshold, side="right")
        fn_count = np.searchsorted(obs_with_np, threshold, side="right")

        _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
        _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
        if not (0 < alpha_upper < 1 and 0 < beta_upper < 1):
            continue

        current_eps = max(
            norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper),
            norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper),
        )
        if current_eps > max_eps:
            max_eps = current_eps
            optimal_threshold = float(threshold)
            false_positives = fp_count
            false_negatives = fn_count

    fpr = false_positives / T_audit
    fnr = false_negatives / T_audit

    print("\n--- Audit Analysis Results ---", file=f)
    print(f"Optimal Decision Threshold (Advantage Score) found: {optimal_threshold:.4f}", file=f)
    print(f"Corresponding μ: {max_eps:.4f}", file=f)
    print(f"  False Positive Rate (α) = {fpr:.4f} ({false_positives}/{T_audit})", file=f)
    print(f"  False Negative Rate (β) = {fnr:.4f} ({false_negatives}/{T_audit})", file=f)

    _, alpha_upper = clopper_pearson(false_positives, T_audit, confidence)
    _, beta_upper = clopper_pearson(false_negatives, T_audit, confidence)
    print(f"\n{confidence*100:.0f}% CI upper bound for α: {alpha_upper:.4f}", file=f)
    print(f"{confidence*100:.0f}% CI upper bound for β: {beta_upper:.4f}", file=f)

    if not (0 < alpha_upper < 1 and 0 < beta_upper < 1):
        print("\nCould not compute a meaningful μ value.", file=f)
        return

    mu_lower_bound = norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)
    print(f"\nEmpirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)


def procedure(
    output_filename: str,
    num_subsets: int,
    all_texts: List[str],
    llm_classifier,
    *,
    num_exemplars_per_subsets: int,
    delta: float,
    T_collect: int,
    T_simulation: int,
    epsilons: Sequence[float],
    confidence: float,
):
    with open(output_filename, "w", encoding="utf-8") as f:
        print("=" * 20 + " Membership Inference f-DP Privacy Audit " + "=" * 20, file=f)

        # Binary labels for membership verification
        class_labels = ["Yes", "No"]

        if len(all_texts) <= num_subsets*num_exemplars_per_subsets + 1:
            raise ValueError("Not enough texts to sample a canary and base exemplars.")

        # sample by index to handle duplicates robustly
        n = len(all_texts)
        canary_idx = random.randrange(n)
        canary_record_text = all_texts[canary_idx]

        remaining_indices = [i for i in range(n) if i != canary_idx]
        base_indices = random.sample(remaining_indices, num_subsets * num_exemplars_per_subsets)
        base_texts = [all_texts[i] for i in base_indices]

        canary_class_label = "Yes"
        base_exemplars = [(t, "No") for t in base_texts]

        exemplars_without_target = base_exemplars
        exemplars_with_target = [(canary_record_text, "Yes")] + base_exemplars

        print(f"Randomly selected canary for audit: '{canary_record_text[:100]}...'", file=f)

        # placeholder epsilon for collection stage (sigma not used in clean vector collection)
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=1.0,
            delta=delta,
        )

        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=dp_classifier,
            exemplars_with=exemplars_with_target,
            exemplars_without=exemplars_without_target,
            query=canary_record_text,
            num_subsets=num_subsets,
            T_collect=T_collect,
            f=f,
        )

        print(f"\nNumber of Subsets (K): {num_subsets}", file=f)
        print(f"\nCollected {len(collected_vectors_with)} vectors for 'With Canary' empirical distribution.", file=f)
        print(f"Collected {len(collected_vectors_without)} vectors for 'Without Canary' empirical distribution.", file=f)

        for epsilon in epsilons:
            f.write(f"\n{'='*15} ANALYSIS for ε={epsilon} {'='*15}\n")
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier,
                class_labels=class_labels,
                epsilon=float(epsilon),
                delta=delta,
            )

            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms via bootstrap resampling for ε={epsilon} ---")
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
                confidence=confidence,
            )

    print(f"\nMembership inference audit finished. Results saved to '{output_filename}'.")


# ==============================================================================
# 5. Main (argparse)
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="DP-ICL white-box bootstrap audit (dataset-agnostic: text-only).")

    # dataset (text-only)
    p.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/ag_news_csv/test.csv",
                   help="Path to dataset file (csv or txt). This script only uses text content.")
    p.add_argument("--data_format", type=str, default="csv", choices=["csv", "txt"],
                   help="Input format. csv reads a column; txt reads one sample per line.")
    p.add_argument("--text_col", type=int, default=2, help="0-indexed text column for CSV input.")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter.")
    p.add_argument("--no_header", action="store_true", help="Set if the CSV has NO header row.")
    p.add_argument("--encoding", type=str, default="utf-8", help="File encoding.")
    p.add_argument("--max_rows", type=int, default=0, help="Optional cap on number of loaded rows (0 = no cap).")

    # vLLM/OpenAI endpoint
    p.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    p.add_argument("--api_key", type=str, default="EMPTY", help="API key (often dummy for local vLLM).")
    p.add_argument("--model_id", type=str, default="/local/scratch/yxia327/gpt-oss-20b", help="Model id for chat.completions.")
    p.add_argument("--system_prompt", type=str,
                   default="Reasoning: low. No analysis nor commentary needed. Just answer Yes or No.",
                   help="System prompt for the verifier LLM.")
    p.add_argument("--max_tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.0)

    # audit parameters
    p.add_argument("--num_subsets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14])
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--num_exemplars_per_subsets", type=int, default=2)
    p.add_argument("--T_collect", type=int, default=200)
    p.add_argument("--T_simulation", type=int, default=400000)
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--confidence", type=float, default=0.95)

    # output & reproducibility
    p.add_argument("--output_dir", type=str, default="experiment_results_white")
    p.add_argument("--output_prefix", type=str, default="gpt-run")
    p.add_argument("--seed", type=int, default=0, help="0 means do not force a global seed.")

    return p.parse_args()


def main():
    args = parse_args()

    print("NOTE: This script is dataset-agnostic and ONLY uses text content (no labels).")
    print(f"Data: {args.data_path} (format={args.data_format}, text_col={args.text_col})")

    max_rows = None if args.max_rows == 0 else args.max_rows
    all_texts = load_texts(
        data_path=args.data_path,
        data_format=args.data_format,
        text_col=args.text_col,
        delimiter=args.delimiter,
        has_header=(not args.no_header),
        encoding=args.encoding,
        max_rows=max_rows,
    )
    if len(all_texts) < args.num_exemplars_per_subsets * args.num_subsets + 2:
        raise ValueError(f"Loaded only {len(all_texts)} texts; need at least {args.num_exemplars_per_subsets * args.num_subsets + 2}.")

    os.makedirs(args.output_dir, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    llm_classifier = make_llm_classifier(
        client=client,
        model_id=args.model_id,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    for num_subsets in args.num_subsets:
        for run_id in range(1, args.num_runs + 1):
            if args.seed and args.seed != 0:
                _maybe_set_seed(args.seed + run_id)

            output_filename = os.path.join(args.output_dir, f"{args.output_prefix}{run_id}-subsets{num_subsets}.txt")
            procedure(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_texts=all_texts,
                llm_classifier=llm_classifier,
                num_exemplars_per_subsets=args.num_exemplars_per_subsets,
                delta=args.delta,
                T_collect=args.T_collect,
                T_simulation=args.T_simulation,
                epsilons=args.epsilons,
                confidence=args.confidence,
            )

if __name__ == "__main__":
    main()
