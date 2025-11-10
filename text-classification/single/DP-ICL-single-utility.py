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

# 假设新的 utility_test_procedure 函数
def utility_test_procedure(output_filename, num_subsets, all_data, llm_classifier_func):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " DP-ICL Classification Utility Test " + "="*20, file=f)
        class_labels = ["World", "Sports", "Business", "Sci/Tech"]
        
        # --- 1. 定义数据和设置 ---
        # 基础 exemplars (ICL 训练集)
        base_exemplars_data = all_data[1:31]
        base_exemplars = [(rec[2], class_labels[int(rec[0]) - 1]) for rec in base_exemplars_data]
        
        # Mislabeled exemplar (Poisoned data point)
        canary_record_text = "Real Madrid has signed a new player."
        mislabeled_label = "Business"
        canary_exemplar_mislabeled = (canary_record_text, mislabeled_label)
        
        # 定义两种 ICL 设置
        exemplars_with_mislabeled = [canary_exemplar_mislabeled] + base_exemplars
        exemplars_without_mislabeled = base_exemplars
        
        # --- 2. 选取测试集 ---
        # 选取第 500 个样本到第 1000 个样本作为测试集 (共 500 个)
        test_data_start_index = 500
        test_data_end_index = 1000
        test_set = all_data[test_data_start_index:test_data_end_index] 
        T_samples = len(test_set)
        
        print(f"Number of Subsets (K): {num_subsets}", file=f)
        print(f"Test Samples (T): {T_samples}", file=f)
        
        # --- 3. 循环测试不同 epsilon 值 ---
        for epsilon in [1.0, 2.0, 4.0, 8.0]:
            print(f"\n--- Testing Epsilon (ε) = {epsilon} ---", file=f)
            dp_classifier = DPICLForClassification(
                llm_callable=llm_classifier_func,
                class_labels=class_labels,
                epsilon=epsilon,
                delta=1e-5
            )

            # --- 4. 运行实用性测试 (With Mislabeled Exemplar) ---
            acc_with, total_with = _run_utility_test(
                dp_classifier, exemplars_with_mislabeled, test_set, num_subsets, class_labels)
            
            # --- 5. 运行实用性测试 (Without Mislabeled Exemplar) ---
            acc_without, total_without = _run_utility_test(
                dp_classifier, exemplars_without_mislabeled, test_set, num_subsets, class_labels)

            # --- 6. 打印结果 ---
            print(f"Accuracy (With Mislabeled) = {acc_with/total_with:.4f} ({acc_with}/{total_with})", file=f)
            print(f"Accuracy (Without Mislabeled) = {acc_without/total_without:.4f} ({acc_without}/{total_without})", file=f)
            
            # 简单计算 Utility Drop (实用性损失)
            utility_drop = (acc_without/total_without) - (acc_with/total_with)
            print(f"Utility Drop = {utility_drop:.4f}", file=f)

    print(f"\nUtility Test finished. Results saved to '{output_filename}'")

# --- 新增的辅助函数 ---
def _run_utility_test(dp_classifier, exemplars, test_set, num_subsets, class_labels):
    correct_count = 0
    total_count = 0
    for rec in tqdm(test_set, desc=f"Testing ε={dp_classifier.epsilon}, {'With' if len(exemplars) > len(test_set[0]) else 'Without'}"):
        query_article = rec[2]
        true_label = class_labels[int(rec[0]) - 1]
        
        # Call the DP-ICL mechanism once to get the final noisy prediction
        predicted_label, _ = dp_classifier(
            exemplars=exemplars, 
            query_str=query_article, 
            num_subsets=num_subsets
        )
        
        if predicted_label == true_label:
            correct_count += 1
        total_count += 1
        
    return correct_count, total_count


# --- 更新主执行块 ---
if __name__ == '__main__':

    try:
        with open("/local/scratch/yxia327/data/ag_news_csv/test.csv", 'r', encoding='utf-8') as f:
            # 确保你的数据足够，至少需要 1000 个样本 (30 ICL + 970 备用)
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print("Error: `ag_news_csv/test.csv` not found.")
        exit()

    # 清除旧的审计函数，调用新的实用性测试函数
    alfa = [2,4,6,8,10,12,14]
    for num_subsets in alfa:
        # 只需运行一次，因为不再是统计性的隐私审计
        output_filename = f'experiment_results_single/utility-test-subsets{num_subsets}.txt'
        # 注意: 需要将 procedure 替换为 utility_test_procedure
        utility_test_procedure(output_filename, num_subsets, all_data, llm_classifier)