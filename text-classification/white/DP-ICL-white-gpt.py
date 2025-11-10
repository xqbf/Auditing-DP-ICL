import numpy as np
import math
import csv
from scipy.stats import beta, norm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import random
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# ==============================================================================
# 1. Hugging Face Transformers Setup (本地模型配置)
# ==============================================================================
# local_model_path = "/local/scratch/yxia327/gpt-oss-20b"

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# print("Loading model...")
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# lm = dspy.LM("openai/gpt-oss-20b",
#              api_base="http://localhost:8000/v1",  # ensure this points to your port
#              api_key="local", model_type="chat")
# dspy.configure(lm=lm)

def llm_classifier(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
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

    # Make the API call WITHOUT the extra_body parameter
    resp = client.chat.completions.create(
        model="/local/scratch/yxia327/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Reasoning: low. No analysis nor commentary needed. Just answer Yes or No."},
            {"role": "user", "content": prompt_content}
            ],
        max_tokens=1000,
        temperature=0.0,
        # reasoning={"effort": "disabled"}
    )

    # print(resp)

    response_message = resp.choices[0].message
    output = None

    # 检查 message 对象和它的 content 字段是否存在
    if response_message and response_message.content:
        # 如果 content 存在，才进行后续处理
        response = response_message.content.strip()

        if response.lower().startswith("yes"):
            output = "Yes" 
        elif response.lower().startswith("no"):
            output = "No"
        else:
            # 如果 content 存在但不符合预期 (例如返回 "I cannot determine")，执行后备方案
            print(f"!!! 警告: 模型返回了非预期的内容: '{response}'")
            output = random.choice(exemplars_labels)
    else:
        # 如果 content 是 None，直接执行后备方案
        print("!!! 警告: 模型返回的 content 为 None，使用随机标签。")
        output = random.choice(exemplars_labels)

    # --- END: 修改后的健壮性检查 ---

    return output

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
# 5. Mean Vector Simulation Workflow - MODIFIED
# ==============================================================================

def collect_vote_vectors_for_resampling(dp_classifier, exemplars_with, exemplars_without, query, num_subsets, T_collect, f):
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
    STAGE 3: Run f-DP statistical analysis on the simulated histograms. (Fast & MODIFIED)
    """
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
    
    # --- ADDED: Calculate mu for a single threshold at 0 ---
    single_max_eps = -float('inf')
    fp_count_at_zero = T_audit - np.searchsorted(obs_without_np, 0, side='right')
    fn_count_at_zero = np.searchsorted(obs_with_np, 0, side='right')
    _, alpha_upper_at_zero = clopper_pearson(fp_count_at_zero, T_audit, confidence)
    _, beta_upper_at_zero = clopper_pearson(fn_count_at_zero, T_audit, confidence)
    if alpha_upper_at_zero < 1 and beta_upper_at_zero < 1 and alpha_upper_at_zero > 0 and beta_upper_at_zero > 0:
        single_max_eps =  max(norm.ppf(1 - beta_upper_at_zero) - norm.ppf(alpha_upper_at_zero), norm.ppf(1 - alpha_upper_at_zero) - norm.ppf(beta_upper_at_zero))
    print(f"\nSingle threshold at 0 gives μ: {single_max_eps:.4f}", file=f)
    
    print("Finding optimal decision threshold...")
    for threshold in tqdm(candidate_thresholds, desc="Finding Threshold"):
        fp_count = T_audit - np.searchsorted(obs_without_np, threshold, side='right')
        fn_count = np.searchsorted(obs_with_np, threshold, side='right')
        _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
        _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
        if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0: continue
        # --- MODIFIED: Use max for symmetric calculation ---
        current_eps = max(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper), norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper))
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
        
        # --- MODIFIED: Randomly sample data for the audit ---
        num_base_exemplars = 30  # Number of non-canary records
        
        if len(all_data) <= num_base_exemplars:
            raise ValueError("Not enough data to sample a canary and base exemplars.")

        # 1. Randomly select a record from the dataset to be the canary
        canary_record = random.choice(all_data)
        canary_record_text = canary_record[2]
        
        # 2. Randomly select base exemplars from the rest of the data
        other_data = [rec for rec in all_data if rec != canary_record]
        base_exemplars_data = random.sample(other_data, num_base_exemplars)
        
        # 3. Prepare adjacent datasets
        canary_class_label = "Yes"
        base_exemplars = [(record[2], "No") for record in base_exemplars_data]
        
        # Dataset WITHOUT the target member
        exemplars_without_target = base_exemplars
        # Dataset WITH the target member (add-one setup)
        exemplars_with_target = [(canary_record_text, "Yes")] + base_exemplars
        
        print(f"Randomly selected canary for audit: '{canary_record_text[:100]}...'", file=f)
        
        # --- DP-ICL Instantiation ---
        # A temporary classifier instance for the collection stage. 
        # The epsilon value is reset in the loop for the analysis stage.
        dp_classifier = DPICLForClassification(
            llm_callable=llm_classifier,
            class_labels=class_labels,
            epsilon=1.0,  # Placeholder, not used for clean vector collection
            delta=1e-5
        )
        
        # --- Execute Three-Stage Audit with Bootstrap ---
        
        # Stage 1: Collect clean vote vectors to build empirical distributions
        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=dp_classifier,
            exemplars_with=exemplars_with_target,
            exemplars_without=exemplars_without_target,
            query=canary_record_text,
            num_subsets=num_subsets,
            T_collect=2000,  # Number of runs to establish the empirical distribution
            f=f # Pass file handle for logging
        )
        print(f"\n Number of Subsets (K): {num_subsets}", file=f)
        print(f"\nCollected {len(collected_vectors_with)} vectors for 'With Canary' empirical distribution.", file=f)
        print(f"Collected {len(collected_vectors_without)} vectors for 'Without Canary' empirical distribution.", file=f)
        
        # Stage 2 & 3: Simulate by bootstrapping and run analysis for each epsilon
        for epsilon in [1.0, 2.0, 4.0, 8.0]:
            f.write(f"\n{'='*15} ANALYSIS for ε={epsilon} {'='*15}\n")
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=1e-5
            )
            T_simulation = 400000
            print(f"\n--- STAGE 2: Simulating {T_simulation} histograms via bootstrap resampling for ε={epsilon} ---")
            histograms_with = simulate_histograms_from_bootstrap(
                collected_vectors_with, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)
            histograms_without = simulate_histograms_from_bootstrap(
                collected_vectors_without, dp_classifier.sigma, dp_classifier.num_classes, T_simulation)

            # Stage 3: Run f-DP analysis
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
    
    
    alfa = [2,4,6,8,10,12,14]
    for num_subsets in alfa:
        for run_id in range(1,6): # 5 repeticiones por configuración
            output_filename = f'experiment_results_white/gpt-run{run_id}-subsets{num_subsets}.txt'
            procedure(output_filename, num_subsets, all_data, llm_classifier)