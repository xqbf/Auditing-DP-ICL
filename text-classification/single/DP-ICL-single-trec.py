import numpy as np
import math
import csv
from scipy.stats import beta, norm
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
from tqdm import tqdm

# ==============================================================================
# 0. 设置随机种子以保证结果可复现
# ==============================================================================
def set_seed(seed: int):
    """
    设置所有相关的随机种子，以确保实验的可复现性。
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
# 1. Hugging Face Transformers Setup (本地模型配置)
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
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ==============================================================================
# 2. LLM Classifier for Membership Inference (Yes/No Task)
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
# 3. <<< REFACTORED: DP-ICL Implementation for Auditing >>>
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
        """Runs the mechanism and returns the pre-noise vote histogram for auditing."""
        votes = self._get_votes(exemplars, query_str, num_subsets)
        histogram = np.zeros(self.num_classes)
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


def collect_vote_vectors_for_resampling(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
    """
    STAGE 1: Collect a set of clean vote vectors to build an empirical distribution. (Expensive)
    """
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors to build empirical distribution ---")
    
    vectors_without = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets)
        vectors_without.append(clean_vector)

    vectors_with = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        vectors_with.append(clean_vector)
    
    # Return the full lists of collected vectors, not their means.
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
    return np.array(histograms)

def get_mean_vote_vectors(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors to compute mean ---")
    
    vectors_without = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets)
        vectors_without.append(clean_vector)
    mean_vector_without = np.mean(vectors_without, axis=0)

    vectors_with = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        vectors_with.append(clean_vector)
    mean_vector_with = np.mean(vectors_with, axis=0)
    
    return mean_vector_with, mean_vector_without

def simulate_histograms_from_mean(mean_vector, sigma, num_classes, T_simulation):
    histograms = []
    for _ in range(T_simulation):
        noisy_hist = mean_vector + np.random.normal(0, sigma, num_classes)
        histograms.append(noisy_hist)
    return np.array(histograms)

def run_audit_with_fixed_threshold_and_fdp_calc(dp_classifier, histograms_with, histograms_without, canary_class_label, f):
    """
    STAGE 3: Uses a fixed threshold to find TPR/FPR, then applies f-DP formulas.
    """
    print(f"\n--- STAGE 3: Running analysis on {len(histograms_with)} simulated histograms ---", file=f)
    print("Method: Fixed threshold (Yes if score_yes >= score_no) with f-DP bound calculation.", file=f)
    
    canary_class_id = dp_classifier.label_to_id[canary_class_label]
    non_canary_class_id = 1 - canary_class_id
    T_audit = len(histograms_with)
    confidence = 0.95

    # Count True Positives: "with" set where score_yes >= score_no
    tp_count = np.sum(histograms_with[:, canary_class_id] >= histograms_with[:, non_canary_class_id])
    
    # Count False Positives: "without" set where score_yes >= score_no
    fp_count = np.sum(histograms_without[:, canary_class_id] >= histograms_without[:, non_canary_class_id])
    
    # Calculate False Negatives
    fn_count = T_audit - tp_count

    tpr = tp_count / T_audit
    fpr = fp_count / T_audit
    fnr = fn_count / T_audit

    print("\n--- Empirical Rates (Fixed Threshold) ---", file=f)
    print(f"True Positive Rate (TPR)  = {tpr:.4f} ({tp_count}/{T_audit})", file=f)
    print(f"False Positive Rate (FPR) = {fpr:.4f} ({fp_count}/{T_audit})", file=f)
    print(f"False Negative Rate (FNR) = {fnr:.4f} ({fn_count}/{T_audit})", file=f)

    # Get Clopper-Pearson confidence interval upper bounds
    _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
    _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)

    print(f"\n{confidence*100}% CI upper bound for α (FPR): {alpha_upper:.4f}", file=f)
    print(f"{confidence*100}% CI upper bound for β (FNR): {beta_upper:.4f}", file=f)

    if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0:
        print("\nCould not compute a meaningful μ value due to invalid CI bounds (0 or 1).", file=f)
        return
        
    # Calculate mu in both directions and take the max, as requested
    mu_lower_bound1 = norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper)
    mu_lower_bound2 = norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)
    mu_lower_bound = max(mu_lower_bound1, mu_lower_bound2)

    print("\n--- f-DP Bound Calculation ---", file=f)
    print(f"Empirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)


def procedure(output_filename, num_subsets, all_data, llm_classifier):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " Membership Inference f-DP Privacy Audit " + "="*20, file=f)
        
        # --- Define the new binary class labels ---
        class_labels = ["Yes", "No"]
        
        # --- Define Adjacent Datasets for Membership Inference ---
        canary_record_text = "At Microsoft Education, we believe every student deserves the opportunity to thrive. Make math accessible for learners with powerful, inclusive tools designed to unlock potential and build confidence to make math fun."
        canary_class_label = "Yes"
        base_exemplars_data = all_data[1:31]
        base_exemplars = [(record[2], "No") for record in base_exemplars_data]
        exemplars_without_target = base_exemplars
        exemplars_with_target = [(canary_record_text, "Yes")] + base_exemplars
        epsilon = 1
        # --- DP-ICL Instantiation ---
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=epsilon,
            delta=1e-5
        )
        

        # --- <<< MODIFIED: Execute Three-Stage Audit with Bootstrap >>> ---
        
        # Stage 1: Collect clean vote vectors to build empirical distributions
        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=dp_classifier,
            exemplars_with=exemplars_with_target,
            exemplars_without=exemplars_without_target,
            query=canary_record_text,
            num_subsets=num_subsets,
            T_collect=2000  # Number of runs to establish the empirical distribution
        )
        print(f"\n Number of Subsets (K): {num_subsets}", file=f)
        print(f"\nCollected {len(collected_vectors_with)} vectors for 'With Canary' empirical distribution.", file=f)
        print(f"Collected {len(collected_vectors_without)} vectors for 'Without Canary' empirical distribution.", file=f)
        
        # Stage 2: Simulate by bootstrapping from the collected vectors
        for epsilon in [0.1,0.5]: # Puedes ajustar esta lista para múltiples valores de epsilon
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=1e-5
            )
            print(f"Theoretical ε={dp_classifier.epsilon}, δ={dp_classifier.delta}", file=f)
            T_simulation = 400000
            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms via bootstrap resampling ---")
            histograms_with = simulate_histograms_from_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
            histograms_without = simulate_histograms_from_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

            # Stage 3: Run f-DP analysis (call remains the same)
            run_audit_with_fixed_threshold_and_fdp_calc(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label,
                f=f
            )
        
    print(f"\nMembership inference audit finished. Results saved to '{output_filename}'.")



# ==============================================================================
# 5. <<< MODIFIED: Main Execution with Three-Stage Audit >>>
# ==============================================================================
if __name__ == '__main__':
    # set_seed(327)
    # --- Data and Task Setup ---
    try:
        with open("/local/scratch/yxia327/data/TREC/test.csv", 'r', encoding='utf-8') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `TREC/test.csv` not found.")
        exit()

    
    alfa = [2,4,6,8,10,12,14]
    for run_id in range(1,6): # 5 repeticiones por configuración
        for num_subsets in alfa:
            output_filename = f'experiment_results_single/trec-run{run_id}-subsets{num_subsets}.txt'
            procedure(output_filename, num_subsets, all_data, llm_classifier)