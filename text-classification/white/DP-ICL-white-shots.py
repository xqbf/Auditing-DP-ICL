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
# ... (此部分代码无任何改动) ...
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

# ==============================================================================
# 4. Privacy Audit Helper Functions - Unchanged
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

# ==============================================================================
# 5. Mean Vector Simulation Workflow - Unchanged
# ... (此部分代码无任何改动) ...
# ==============================================================================
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
def procedure(output_filename, num_subsets, all_data, llm_classifier,num_shots):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " Membership Inference f-DP Privacy Audit " + "="*20, file=f)
        
        # --- Define the binary class labels for the membership inference task ---
        class_labels = ["Yes", "No"]
        
        # --- Randomly sample data for the audit ---
        num_base_exemplars = num_shots*num_subsets  # Same number as the original fixed slice all_data[1:31]
        
        if len(all_data) <= num_base_exemplars:
            raise ValueError("Not enough data to sample a canary and base exemplars.")

        # 1. Randomly select a record from the real dataset to be the canary
        canary_record = random.choice(all_data)
        canary_record_text = canary_record[2] # The text content is the canary
        
        # 2. Randomly select base exemplars from the rest of the data
        other_data = [rec for rec in all_data if rec != canary_record]
        base_exemplars_data = random.sample(other_data, num_base_exemplars)
        
        # 3. Prepare adjacent datasets using "Yes"/"No" TASK LABELS
        canary_class_label = "Yes" # The canary's label for this task is always "Yes"
        
        # Base exemplars for the audit task are always labeled "No"
        base_exemplars = [(record[2], "No") for record in base_exemplars_data]
        
        # Dataset WITHOUT the target member
        exemplars_without_target = base_exemplars
        # Dataset WITH the target member
        exemplars_with_target = [(canary_record_text, "Yes")] + base_exemplars[1:]
        
        print(f"Randomly selected canary for audit: '{canary_record_text[:100]}...'", file=f)
        
        # --- DP-ICL Instantiation for the audit task ---
        # The epsilon value will be set inside the loop
        
        # --- Execute Three-Stage Audit with Bootstrap ---
        
        # Stage 1: Collect clean vote vectors
        # A temporary classifier instance is needed for this step
        temp_dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=1.0, # Epsilon value here doesn't affect the clean vector collection
            delta=1e-5
        )
        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=temp_dp_classifier,
            exemplars_with=exemplars_with_target,
            exemplars_without=exemplars_without_target,
            query=canary_record_text,
            num_subsets=num_subsets,
            T_collect=200,
            f=f
        )
        print(f"\nNumber of Subsets (K): {num_subsets}", file=f)
        print(f"Collected {len(collected_vectors_with)} vectors for 'With Canary' distribution.", file=f)
        print(f"Collected {len(collected_vectors_without)} vectors for 'Without Canary' distribution.", file=f)
        
        # Stages 2 & 3: Simulate and run f-DP analysis for each epsilon
        for epsilon in [1.0, 2.0, 4.0, 8.0]: 
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=1e-5
            )
            print(f"\n--- Analyzing for Theoretical ε={dp_classifier.epsilon} ---")
            f.write(f"\n{'='*15} ANALYSIS for ε={epsilon} {'='*15}\n")

            T_simulation = 400000
            histograms_with = simulate_histograms_from_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
            histograms_without = simulate_histograms_from_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

            run_fdp_audit_on_histograms(
                dp_classifier=dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label=canary_class_label,
                f=f
            )
        
    print(f"\nMembership inference audit finished. Results saved to '{output_filename}'.")

# ==============================================================================
# 7. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # --- Data Setup ---
    try:
        with open("/local/scratch/yxia327/data/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found.")
        exit()
    
    alfa = [1,2,4,8]
    for num_subsets in [4]:
        for num_shots in alfa:
            for run_id in range(1, 3):
                output_filename = f'experiment_results_white/shots-run{run_id}-subsets{num_subsets}-shots{num_shots}.txt'
                # Each call to procedure will now use a new random canary and base exemplars
                procedure(output_filename, num_subsets, all_data, llm_classifier, num_shots=num_shots)