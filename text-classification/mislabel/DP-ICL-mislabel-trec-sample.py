import numpy as np
import math
import csv
from scipy.stats import beta, norm
import torch
import sys
import random
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# ==============================================================================
# 0. Set Seed for Reproducibility
# ... (此部分代码无任何改动) ...
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
# 1. Hugging Face Transformers Setup
# ... (此部分代码无任何改动) ...
# ==============================================================================
local_model_path = "/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print(f"模型已成功加载到设备: {model.device}")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ==============================================================================
# 2. LLM Classifier for Normal Classification
# ... (此部分代码无任何改动) ...
# ==============================================================================
def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    使用本地 Hugging Face Transformers 模型进行正常的文本分类。
    """
    formatted_exemplars = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_exemplars += f"Text: {text}\nLabel: {label}\n\n"
    
    # Updated prompt for a standard classification task
    prompt_content = f"""You are a highly accurate text classifier. Classify the following text into one of the provided categories based on the examples.
The possible labels are: DESC, ENTY, ABBR, HUM, NUM, LOC.  Only output the label.

Examples:
{formatted_exemplars}

Query Text: {query_article}
Label:"""
    messages = [{"role": "user", "content": prompt_content}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(max_new_tokens=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    outputs = model.generate(input_ids, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


    # print(prompt_content)
    # print(response)

    # Update valid labels and parsing logic
    valid_labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
    # Simple check for the first word of the response to match a label
    for label in valid_labels:
        if response.startswith(label):
            return label
    return np.random.choice(valid_labels) # Return a random label if parsing fails

# ==============================================================================
# 3. DP-ICL Implementation for Auditing
# ... (此部分代码无任何改动) ...
# ==============================================================================
class DPICLForClassification:
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
        votes = self._get_votes(exemplars, query_str, num_subsets)
        histogram = np.zeros(self.num_classes)
        for vote in votes:
            if vote in self.label_to_id:
                histogram[self.label_to_id[vote]] += 1
        return histogram

    def __call__(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> tuple[str, np.ndarray]:
        clean_histogram = self.get_clean_vote_vector(exemplars, query_str, num_subsets)
        noisy_histogram = clean_histogram + np.random.normal(0, self.sigma, self.num_classes)
        winner_index = np.argmax(noisy_histogram)
        return (self.id_to_label[winner_index], noisy_histogram)

# ==============================================================================
# 4. Privacy Audit Helper Functions
# ... (此部分代码无任何改动) ...
# ==============================================================================
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

def collect_vote_vectors(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors ---")
    vectors_without = [dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")]
    vectors_with = [dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")]
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
    # ... (此函数无任何改动) ...
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
# <<< MODIFIED: Procedure function with randomized data preparation for TREC >>>
# ==============================================================================
def procedure(output_filename, num_subsets, all_data, llm_classifier_func):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " DP-ICL Membership Inference Mislabeled Audit (TREC) " + "="*20, file=f)
        
        class_labels = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

        # --- Randomly Sample Canary and Base Exemplars ---
        num_base_exemplars = 2*num_subsets
        if len(all_data) <= num_base_exemplars:
            raise ValueError("Not enough data to sample for the audit.")

        # 1. Randomly select a canary from the TREC dataset
        # all_data is already in (text, label) format from the main block
        canary_record = random.choice(all_data)
        canary_record_text = canary_record[0]
        
        # 2. Dynamically determine its true and a mislabeled label
        true_label = canary_record[1]
        possible_mislabels = [label for label in class_labels if label != true_label]
        mislabeled_label = random.choice(possible_mislabels)
        
        # 3. For the audit, the canary exemplar is paired with its TRUE label
        canary_exemplar_labeled = (canary_record_text, true_label)
        # The audit goal is to infer the presence of the correctly labeled canary
        canary_class_label = true_label
        
        # 4. The query simulates an attacker who knows the text but guesses a WRONG label
        query_text = f"Query Text: {canary_record_text}\nLabel: {mislabeled_label}\n"*6 + \
                     f"Query Text: {canary_record_text}\nLabel:"
        
        # 5. Randomly select base exemplars from the rest of the data
        other_data = [rec for rec in all_data if rec != canary_record]
        base_exemplars = random.sample(other_data, num_base_exemplars)

        # 6. Create the two adjacent datasets for the audit
        exemplars_with_labeled = [canary_exemplar_labeled] + base_exemplars[1:]
        exemplars_without_labeled = base_exemplars

        # --- Start Audit ---
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier_func,
            class_labels=class_labels,
            epsilon=1,
            delta=1e-5
        )
        
        print(f"Random Canary Text: '{canary_record_text[:100]}...'", file=f)
        print(f"Canary True Label: {true_label}, Mislabeled as: {mislabeled_label}", file=f)
        
        # Stage 1: Collect raw vote vectors
        collected_vectors_with, collected_vectors_without = collect_vote_vectors(
            dp_classifier=dp_classifier,
            exemplars_with=exemplars_with_labeled,
            exemplars_without=exemplars_without_labeled,
            query=canary_record_text, # Using the raw text as query for the classifier
            num_subsets=num_subsets,
            T_collect=200
        )
        mean_vector_with = np.mean(collected_vectors_with, axis=0) if collected_vectors_with else np.zeros(dp_classifier.num_classes)
        mean_vector_without = np.mean(collected_vectors_without, axis=0) if collected_vectors_without else np.zeros(dp_classifier.num_classes)
        print(f"\n Number of Subsets (K): {num_subsets}", file=f)
        print(f"\nComputed Mean Vector (With Canary): {mean_vector_with}", file=f)
        print(f"Computed Mean Vector (Without Canary): {mean_vector_without}", file=f)
        for epsilon in [0.1, 1.0, 2.0, 4.0, 8.0]:
            dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier_func,
            class_labels=class_labels,
            epsilon=epsilon,
            delta=1e-5
            )
            # Stage 2: Simulate by bootstrapping and adding noise
            T_simulation = 400000
            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms using BOOTSTRAP RESAMPLING ---")
            print(f"Theoretical ε={dp_classifier.epsilon}, δ={dp_classifier.delta}", file=f)
            histograms_with = simulate_histograms_with_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
            histograms_without = simulate_histograms_with_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

            # Stage 3: Run analysis
            run_audit_with_fixed_threshold_and_fdp_calc(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label, # The TRUE label
                f=f
            )
    print(f"\nAudit finished. Results saved to '{output_filename}'")


# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == '__main__':

    try:
        import pandas as pd
        # 读取 TREC 测试集
        df = pd.read_csv("/local/scratch/yxia327/data/TREC/test.csv")
        all_data_raw = df.iloc[1:].values.tolist()  # 去掉表头

        # coarse label id -> 六大类
        coarse_to_label = {
            "0": "DESC",
            "1": "ENTY",
            "2": "ABBR",
            "3": "HUM",
            "4": "NUM",
            "5": "LOC"
        }

        # 转换成统一格式 (text, label)
        all_data = [(row[2], coarse_to_label[str(row[0])]) for row in all_data_raw]

    except FileNotFoundError:
        print("Error: `TREC/test.csv` not found.")
        exit()

    
    alfa = [2, 4, 6, 8, 10, 12, 14]
    for num_subsets in alfa:
        for run_id in range(1, 6):
            output_filename = f'experiment_results_mislabel/sample-trec-run{run_id}-subsets{num_subsets}.txt'
            procedure(output_filename, num_subsets, all_data, llm_classifier)