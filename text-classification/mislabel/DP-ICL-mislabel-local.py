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
# 1. Mock LLM Classifier (Placeholder)
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
# 2. LLM Classifier for Membership Inference (Yes/No Task)
# ==============================================================================
# ==============================================================================
# 2. LLM Classifier for Normal Classification
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
The possible labels are: World, Sports, Business, Sci/Tech.  Only output the label.

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
    valid_labels = ["World", "Sports", "Business", "Sci/Tech"]
    # Simple check for the first word of the response to match a label
    for label in valid_labels:
        if response.startswith(label):
            return label
    return np.random.choice(valid_labels) # Return a random label if parsing fails

# ==============================================================================
# 3. DP-ICL Implementation for Auditing
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

# MODIFIED: Returns raw vectors instead of the mean
def collect_vote_vectors(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors ---")
    vectors_without = [dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")]
    vectors_with = [dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")]
    return vectors_with, vectors_without

# NEW: Simulates histograms using bootstrap resampling
def simulate_histograms_with_bootstrap(collected_vectors, sigma, num_classes, T_simulation):
    if not collected_vectors:
        return np.array([])
    collected_vectors_arr = np.array(collected_vectors)
    num_collected = len(collected_vectors_arr)
    # Step 1: Sample indices with replacement
    bootstrap_indices = np.random.randint(0, num_collected, size=T_simulation)
    # Step 2: Create bootstrapped sample of clean histograms
    bootstrapped_clean_histograms = collected_vectors_arr[bootstrap_indices]
    # Step 3: Generate and add Gaussian noise
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

def procedure(output_filename, num_subsets, all_data, llm_classifier_func):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " DP-ICL Membership Inference Mislabeled Audit " + "="*20, file=f)
        class_labels = ["World", "Sports", "Business", "Sci/Tech"]
        canary_record_text = "At Microsoft Education, we believe every student deserves the opportunity to thrive. Make math accessible for learners with powerful, inclusive tools designed to unlock potential and build confidence to make math fun."
        true_label = "Sci/Tech"
        mislabeled_label = "Sports"
        canary_exemplar_labeled = (canary_record_text, true_label)
        query_text = f"Query Text: {canary_record_text}\nLabel: {mislabeled_label}\n"*6+f"Query Text: {canary_record_text}\nLabel:"
        canary_class_label = true_label
        base_exemplars_data = all_data[1:31]
        base_exemplars = [(rec[2], class_labels[int(rec[0]) - 1]) for rec in base_exemplars_data]
        exemplars_with_labeled = [canary_exemplar_labeled] + base_exemplars
        exemplars_without_labeled = base_exemplars
        epsilon = 1
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier_func,
            class_labels=class_labels,
            epsilon=epsilon,
            delta=1e-5
        )
        
        print(f"Canary True Label: {true_label}, Mislabeled as: {mislabeled_label}", file=f)
        
        # Stage 1: Collect raw vote vectors
        collected_vectors_with, collected_vectors_without = collect_vote_vectors(
            dp_classifier=dp_classifier,
            exemplars_with=exemplars_with_labeled,
            exemplars_without=exemplars_without_labeled,
            query=query_text,
            num_subsets=num_subsets,
            T_collect=2000  # Reduced for faster execution
        )
        mean_vector_with = np.mean(collected_vectors_with, axis=0) if collected_vectors_with else np.zeros(dp_classifier.num_classes)
        mean_vector_without = np.mean(collected_vectors_without, axis=0) if collected_vectors_without else np.zeros(dp_classifier.num_classes)
        print(f"\n Number of Subsets (K): {num_subsets}", file=f)
        print(f"\nComputed Mean Vector (With Mislabeled Canary): {mean_vector_with}", file=f)
        print(f"Computed Mean Vector (Without Mislabeled Canary): {mean_vector_without}", file=f)
        
        for epsilon in [0.1, 1.0,2.0,4.0,8.0]: # Puedes ajustar esta lista para múltiples valores de epsilon
            # Stage 2: Simulate by bootstrapping and adding noise
            print(f"Theoretical ε={epsilon}, δ={dp_classifier.delta}", file=f)
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier_func,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=1e-5
            )
            T_simulation = 400000 # Reduced for faster execution
            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms using BOOTSTRAP RESAMPLING ---")
            histograms_with = simulate_histograms_with_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
            histograms_without = simulate_histograms_with_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

            # Stage 3: Run analysis
            run_audit_with_fixed_threshold_and_fdp_calc(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label,
                f=f
            )
    print(f"\nAudit finished. Results saved to '{output_filename}'")

# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == '__main__':

    try:
        with open("/local/scratch/yxia327/data/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found.")
        exit()


    
    alfa = [2,4,6,8,10,12,14]
    for num_subsets in alfa:
        for run_id in range(1,6): # 5 repeticiones por configuración
            output_filename = f'experiment_results_mislabel/mislabel-run{run_id}-subsets{num_subsets}.txt'
            procedure(output_filename, num_subsets, all_data, llm_classifier)