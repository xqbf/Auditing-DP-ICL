import numpy as np
import math
import csv
from scipy.stats import beta, norm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
import sys
from tqdm import tqdm

# ==============================================================================
# 0. Set seed for reproducibility
# ==============================================================================
def set_seed(seed: int):
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
# 2. LLM Classifiers for Different Tasks
# ==============================================================================

# <<< MODIFIED: This classifier remains for the PRIVACY AUDIT part >>>
def llm_classifier_audit(contexts: list[str], query_article: str, exemplars_labels: list[str]) -> str:
    """
    LLM for Membership Inference (Yes/No Task).
    Asks if the query_article is present in the context.
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
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(max_new_tokens=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    if "yes" in response.lower():
        return "Yes"
    elif "no" in response.lower():
        return "No"
    return np.random.choice(["Yes", "No"])

# <<< NEW: A new classifier for the UTILITY TEST part >>>
def llm_classifier_utility(contexts: list[str], query_article: str, exemplars_labels: list[str], class_options: list[str]) -> str:
    """
    LLM for Text Classification (AG News Task).
    Asks the model to classify the query_article based on the examples in the context.
    """
    formatted_exemplars = ""
    for text, label in zip(contexts, exemplars_labels):
        formatted_exemplars += f"Text: {text}\nCategory: {label}\n\n"

    # Join class options for the prompt
    class_options_str = ", ".join(class_options)

    prompt_content = f"""You are an expert text classifier. Based on the examples provided, classify the new query text into one of the following categories: {class_options_str}.
Your response must be only a single word corresponding to the category.

--- EXAMPLES ---
{formatted_exemplars}
--- NEW QUERY ---
Text: {query_article}
Category:
"""
    messages = [{"role": "user", "content": prompt_content}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(max_new_tokens=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    # Find the most likely class label in the response
    for label in class_options:
        if label.lower() in response.lower():
            return label
    return np.random.choice(class_options)

# ==============================================================================
# 3. DP-ICL Implementation (No changes needed here)
# ==============================================================================
class DPICLForClassification:
    def __init__(self, llm_callable, class_labels: list, epsilon: float, delta: float, **llm_kwargs):
        self.llm = llm_callable
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.label_to_id = {label: i for i, label in enumerate(class_labels)}
        self.id_to_label = {i: label for i, label in enumerate(class_labels)}
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = (math.sqrt(2) * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        self.llm_kwargs = llm_kwargs # Store extra arguments for the LLM

    def _get_votes(self, exemplars: list[tuple[str, str]], query_str: str, num_subsets: int) -> list[str]:
        shuffled_indices = np.random.permutation(len(exemplars))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        votes = []
        for subset_indices in partition_indices:
            if len(subset_indices) == 0: continue
            subset_exemplars_text = [exemplars[i][0] for i in subset_indices]
            subset_exemplars_labels = [exemplars[i][1] for i in subset_indices]
            prediction = self.llm(
                contexts=subset_exemplars_text,
                query_article=query_str,
                exemplars_labels=subset_exemplars_labels,
                **self.llm_kwargs # Pass extra arguments
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

# ==============================================================================
# 4. Privacy Audit Helper Functions (No changes needed here)
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
    print(f"\n--- STAGE 1: Collecting {T_collect} clean vote vectors to build empirical distribution ---")
    vectors_without = [dp_classifier.get_clean_vote_vector(exemplars_without, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")]
    vectors_with = [dp_classifier.get_clean_vote_vector(exemplars_with, query, num_subsets) for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")]
    return vectors_with, vectors_without

def simulate_histograms_from_bootstrap(collected_vectors, sigma, num_classes, T_simulation):
    histograms = []
    for _ in range(T_simulation):
        clean_vector = random.choice(collected_vectors)
        noisy_hist = clean_vector + np.random.normal(0, sigma, num_classes)
        histograms.append(noisy_hist)
    return np.array(histograms)

def run_audit_with_fixed_threshold_and_fdp_calc(dp_classifier, histograms_with, histograms_without, canary_class_label, f):
    print(f"\n--- STAGE 3: Running analysis on {len(histograms_with)} simulated histograms ---", file=f)
    canary_class_id = dp_classifier.label_to_id[canary_class_label]
    non_canary_class_id = 1 - canary_class_id
    T_audit = len(histograms_with)
    tp_count = np.sum(histograms_with[:, canary_class_id] >= histograms_with[:, non_canary_class_id])
    fp_count = np.sum(histograms_without[:, canary_class_id] >= histograms_without[:, non_canary_class_id])
    fn_count = T_audit - tp_count
    tpr, fpr, fnr = tp_count / T_audit, fp_count / T_audit, fn_count / T_audit
    print("\n--- Empirical Rates (Fixed Threshold) ---", file=f)
    print(f"True Positive Rate (TPR)  = {tpr:.4f} ({tp_count}/{T_audit})", file=f)
    print(f"False Positive Rate (FPR) = {fpr:.4f} ({fp_count}/{T_audit})", file=f)
    
    _, alpha_upper = clopper_pearson(fp_count, T_audit, 0.95)
    _, beta_upper = clopper_pearson(fn_count, T_audit, 0.95)
    print(f"\n95.0% CI upper bound for α (FPR): {alpha_upper:.4f}", file=f)
    print(f"95.0% CI upper bound for β (FNR): {beta_upper:.4f}", file=f)

    if alpha_upper >= 1 or beta_upper >= 1 or alpha_upper <= 0 or beta_upper <= 0:
        print("\nCould not compute a meaningful μ value.", file=f)
        return
    
    mu_lower_bound = max(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper), norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper))
    print("\n--- f-DP Bound Calculation ---", file=f)
    print(f"Empirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_classifier.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_classifier.delta}: {eps_lower_bound:.4f}", file=f)

# ==============================================================================
# 5. Main Procedure <<< MODIFIED TO BE SWITCHABLE >>>
# ==============================================================================
def procedure(output_filename, num_subsets, all_data, label_map, test_utility_on_canary: bool):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " DP-ICL Membership Inference f-DP Audit & Utility Test " + "="*20, file=f)
        
        # --- Data Preparation ---
        num_base_exemplars = 2*num_subsets
        required_samples = num_base_exemplars + (1 if test_utility_on_canary else 2)
        if len(all_data) < required_samples:
            raise ValueError("Not enough data for sampling.")

        canary_record = random.choice(all_data)
        other_data = [rec for rec in all_data if rec != canary_record]
        base_exemplars_data = random.sample(other_data, num_base_exemplars)
        
        base_exemplars = [(rec[2], label_map[rec[0]]) for rec in base_exemplars_data]
        canary_record_text = canary_record[2]
        
        print("Canary Record:", file=f)
        print(f"  (Class: {label_map[canary_record[0]]}, Content: {canary_record[2]})", file=f)
        print("Base Exemplars:", file=f)
        for rec in base_exemplars_data:
            print(f"  (Class: {label_map[rec[0]]}, Content: {rec[2]})", file=f)
        # <<< NEW: Switchable logic for selecting the utility test sample >>>
        if test_utility_on_canary:
            # 方案C: 使用canary作为效用测试的目标
            utility_test_sample = canary_record
            print("--- Utility Test Target: Canary record itself ---", file=f)
        else:
            # 方案B: 使用一个全新的、独立的样本作为效用测试的目标
            remaining_data = [rec for rec in other_data if rec not in base_exemplars_data]
            if not remaining_data:
                raise ValueError("Not enough unique data to select a separate utility test sample.")
            utility_test_sample = random.choice(remaining_data)
            print("--- Utility Test Target: A new, independent sample ---", file=f)
            
        utility_query_text = utility_test_sample[2]
        utility_true_label = label_map[utility_test_sample[0]]

        print(f"Auditing Canary: (Class: {label_map[canary_record[0]]}, Title: {canary_record[1]})", file=f)
        print(f"Utility Test Sample: (Class: {utility_true_label}, Title: {utility_test_sample[1]})", file=f)
        print(f"Number of Subsets (K): {num_subsets}", file=f)

        # ======================================================
        # PART 1: PRIVACY AUDIT (Setup and Clean Vector Collection)
        # ======================================================
        print("\n--- Running Privacy Audit ---")
        audit_labels = ["Yes", "No"]
        exemplars_for_audit_with = [(canary_record_text, "Yes")] + [(r[0], "No") for r in base_exemplars[1:]]
        exemplars_for_audit_without = [(r[0], "No") for r in base_exemplars]
        audit_dp_classifier_temp = DPICLForClassification(llm_callable=llm_classifier_audit, class_labels=audit_labels, epsilon=1.0, delta=1e-5)
        
        collected_vectors_with, collected_vectors_without = collect_vote_vectors_for_resampling(
            dp_classifier=audit_dp_classifier_temp,
            exemplars_with=exemplars_for_audit_with,
            exemplars_without=exemplars_for_audit_without,
            query=canary_record_text,
            num_subsets=num_subsets,
            T_collect=200
        )

        # ======================================================
        # PART 2: UTILITY TEST (Clean Vector Collection)
        # ======================================================
        print("\n--- Collecting Clean Votes for Utility Test (Averaging over runs) ---")
        utility_labels = list(label_map.values())
        utility_dp_classifier_temp = DPICLForClassification(
            llm_callable=llm_classifier_utility,
            class_labels=utility_labels,
            epsilon=999, delta=1e-5,
            class_options=utility_labels
        )
        
        N_CLEAN_VOTES_AVG = 30  # <<< NEW: Number of runs to average for a robust clean vector
        utility_vote_vectors = []
        
        # <<< NEW: Loop to average out partitioning randomness >>>
        for _ in tqdm(range(N_CLEAN_VOTES_AVG), desc="Averaging Utility Clean Votes"):
            vec = utility_dp_classifier_temp.get_clean_vote_vector(
                exemplars=base_exemplars,
                query_str=utility_query_text,
                num_subsets=num_subsets
            )
            utility_vote_vectors.append(vec)
            
        # Average the collected vectors to get a stable, robust clean vector
        clean_vote_vector_utility = np.mean(utility_vote_vectors, axis=0)

        f.write(f"\nUtility Task Robust Clean Vote Vector (averaged over {N_CLEAN_VOTES_AVG} runs): {clean_vote_vector_utility}\n")

        # ======================================================
        # PART 3: COMBINED ANALYSIS FOR EACH EPSILON
        # ======================================================
        T_simulation_audit = 400000
        N_UTILITY_SIMULATIONS = 100

        for epsilon in [0.1, 1.0, 2.0, 4.0, 8.0]:
            f.write(f"\n{'='*15} ANALYSIS for ε={epsilon} {'='*15}\n")

            # --- Privacy Analysis for this Epsilon ---
            # (这部分代码保持不变)
            audit_dp_classifier = DPICLForClassification(llm_callable=llm_classifier_audit, class_labels=audit_labels, epsilon=epsilon, delta=1e-5)
            histograms_with = simulate_histograms_from_bootstrap(collected_vectors_with, audit_dp_classifier.sigma, audit_dp_classifier.num_classes, T_simulation_audit)
            histograms_without = simulate_histograms_from_bootstrap(collected_vectors_without, audit_dp_classifier.sigma, audit_dp_classifier.num_classes, T_simulation_audit)
            run_audit_with_fixed_threshold_and_fdp_calc(
                dp_classifier=audit_dp_classifier,
                histograms_with=histograms_with,
                histograms_without=histograms_without,
                canary_class_label="Yes",
                f=f
            )

            # --- Robust DP-Utility Analysis for this Epsilon ---
            # (这部分代码使用新的、经过平均的 clean_vote_vector_utility，其他保持不变)
            f.write("\n--- Robust DP-Utility Analysis ---\n")
            utility_dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier_utility,
                class_labels=utility_labels,
                epsilon=epsilon,
                delta=1e-5
            )
            correct_predictions_count = 0
            for _ in range(N_UTILITY_SIMULATIONS):
                noise = np.random.normal(0, utility_dp_classifier.sigma, utility_dp_classifier.num_classes)
                noisy_vote_vector = clean_vote_vector_utility + noise
                predicted_label_id = np.argmax(noisy_vote_vector)
                predicted_label = utility_dp_classifier.id_to_label[predicted_label_id]
                if predicted_label == utility_true_label:
                    correct_predictions_count += 1
            accuracy = correct_predictions_count / N_UTILITY_SIMULATIONS
            
            f.write(f"True Label: {utility_true_label}\n")
            f.write(f"Accuracy over {N_UTILITY_SIMULATIONS} noisy simulations (ε={epsilon}): {accuracy:.2%}\n")
            
        print(f"\nAudit and Utility Test finished. Results saved to '{output_filename}'")

# ==============================================================================
# 6. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    try:
        with open("/local/scratch/yxia327/data/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found.")
        exit()
    label_map = {'1': 'World', '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}
    # <<< NEW: Set your desired utility test strategy here >>>
    # True:  方案C - 测试对canary本身的分类效用 (Specific Utility)
    # False: 方案B - 测试对新样本的分类效用 (General Utility)
    USE_CANARY_FOR_UTILITY_TEST = True # 或者设置为 True

    alfa = [2, 4, 6, 8, 10, 12, 14]
    for num_subsets in alfa:
        for run_id in range(1, 6):
            output_filename = f'experiment_results_single/sample-run{run_id}-subsets{num_subsets}.txt'
            procedure(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_data=all_data,
                label_map=label_map,
                test_utility_on_canary=USE_CANARY_FOR_UTILITY_TEST
            )