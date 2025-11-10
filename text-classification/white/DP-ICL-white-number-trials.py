import numpy as np
import math
import csv
from scipy.stats import beta, norm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import random
# ==============================================================================
# 1. Hugging Face Transformers Setup (本地模型配置)
# ==============================================================================
# Specify the path to your local model
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
# 2. <<< MODIFIED: LLM Classifier for Yes/No Task >>>
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
# 3. DP-ICL Implementation - Refactored for Auditing
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

# ==============================================================================
# 4. Privacy Audit Helper Functions - Unchanged
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

# ==============================================================================
# 5. Mean Vector Simulation Workflow - Unchanged
# ==============================================================================

def collect_vote_vectors_for_resampling(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect):
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
    print(f"Computed Mean Vector (Without Canary): {average_vector_without}")


    vectors_with = []
    average_vector_with = np.zeros(dp_classifier.num_classes)
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets)
        vectors_with.append(clean_vector)
        average_vector_with += clean_vector
    average_vector_with /= T_collect
    print(f"Computed Mean Vector (With Canary): {average_vector_with}")
    
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
        mx = 0
        for j in range(len(hist)):
            if j != canary_class_id: mx = max(mx, hist[j])
        observations_with_target.append(hist[canary_class_id] - mx)
    observations_without_target = []
    for hist in histograms_without:
        mx = 0
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
        current_eps = norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper)
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

        # --- DP-ICL Instantiation ---
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=999,
            delta=1e-5
        )
        print(f"Theoretical ε={dp_classifier.epsilon}, δ={dp_classifier.delta}", file=f)

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
        for num_trials in [100,1000,2000,4000,8000,10000,50000,100000,200000]: # 调整这个列表以进行更多的试验
            T_simulation = num_trials  # Number of simulated histograms for the audit
            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms via bootstrap resampling ---")
            for epsilon in [1.0,2.0,4.0,8.0]: # Puedes ajustar esta lista para múltiples valores de epsilon
                print(f"\nSimulating for ε={epsilon}:", file=f)
                dp_classifier = DPICLForClassification(
                    llm_callable=llm_classifier,
                    class_labels=class_labels,
                    epsilon=epsilon,
                    delta=1e-5
                )
                histograms_with = simulate_histograms_from_bootstrap(
                    collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
                histograms_without = simulate_histograms_from_bootstrap(
                    collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

                # Stage 3: Run f-DP analysis (call remains the same)
                run_fdp_audit_on_histograms(
                    dp_classifier=dp_classifier,
                    histograms_with=histograms_with,
                    histograms_without=histograms_without,
                    canary_class_label=canary_class_label,
                    f=f
                )
        
    print(f"\nMembership inference audit finished. Results saved to '{output_filename}'.")



# ==============================================================================
# 6. <<< MODIFIED: Main Execution Block for Membership Inference Audit >>>
# ==============================================================================
if __name__ == '__main__':
    # --- Data Setup ---
    try:
        with open("/local/scratch/yxia327/data/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found.")
        exit()
    
    
    alfa = [4]
    for num_subsets in alfa:
        for run_id in range(1,6): # 5 repeticiones por configuración
            output_filename = f'experiment_results_white/MIA-run{run_id}-subsets{num_subsets}.txt'
            procedure(output_filename, num_subsets, all_data, llm_classifier)