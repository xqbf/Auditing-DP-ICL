import numpy as np
import math
import csv
from scipy.stats import beta, norm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sentence_transformers import SentenceTransformer
import random
import sys
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

# ==============================================================================
# 0. 设置随机种子以保证结果可复现
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
# 1. Hugging Face Transformers & Embedding Setup (本地模型配置)
# ==============================================================================
local_model_path = "/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF"
embedding_model_name = '/local/scratch/yxia327/bge-large-en-v1.5' 

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

print("Loading LLM model...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading Embedding model...")
embedding_model = SentenceTransformer(embedding_model_name)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# ==============================================================================
# 2. LLM Constrained Response (Y1/Y2 Generation Task)
# ==============================================================================
Y1_TARGET = 'The old clock chimed a forgotten, dusty tune.'
Y2_CONTROL = "FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."
CANARY_FACT_SENTENCE = "Patient A's private record shows diagnosis: 'Unique Atypical Fibrosis of the Liver'. "
CANARY_SEARCH_PHRASE = "Unique Atypical Fibrosis of the Liver"

def llm_constrained_generator(contexts: list[str]) -> str:
    """
    受约束的文本生成，返回 Y1 或 Y2。
    """
    exemplar_context = '\n'.join(contexts) if contexts else ""

    query_instruction = f"""
Analyze the "CONTEXTUAL EXEMPLAR" provided below. Your ONLY function is to output ONE of the two predefined responses: "{Y1_TARGET}"  or "{Y2_CONTROL}". DO NOT output anything else, no explanations, no preamble, and no reasoning.

Check if the specific diagnostic phrase "{CANARY_SEARCH_PHRASE}" is present in the context.

If "{CANARY_SEARCH_PHRASE}" is found, output "{Y1_TARGET}".
If the phrase is NOT found, output "{Y2_CONTROL}".
If the context is empty or ambiguous, randomly output "{Y1_TARGET}" or "{Y2_CONTROL}".

CONTEXTUAL EXEMPLAR:
{exemplar_context}

RESPONSE:
"""
    messages = [{"role": "user", "content": query_instruction}]

    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=400, 
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.pad_token_id, 
        do_sample=True,
        temperature=0.9
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    if Y1_TARGET in response:
        return Y1_TARGET
    elif Y2_CONTROL in response:
        return Y2_CONTROL
    else:
        return random.choice([Y1_TARGET, Y2_CONTROL])
        # # Fallback/Inference
        # if CANARY_SEARCH_PHRASE in exemplar_context:
        #     return Y1_TARGET
        # return Y2_CONTROL

# --- 新增函数: 0-shot LLM 生成公共候选集 ---
def generate_zero_shot_candidates(num_candidates: int, embedding_model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟 ESA 算法生成公共候选句 O_C。
    由于审计是针对 Y1/Y2，我们将 Y1 和 Y2 视为最佳候选句，并添加 N 个随机噪音候选句。
    """
    # random_candidates = [
    #     "A statement about the weather tomorrow.",
    #     "The best color for a car is blue.",
    #     "To travel means to grow and learn new things.",
    #     "Financial markets are complex systems.",
    #     "A historical fact about the Roman Empire.",
    #     # Add more random sentences...
    # ]
    random_candidates = []
    for i in range(num_candidates):
        sentence = llm_constrained_generator("")
        random_candidates.append(sentence)
    candidate_texts = list(set(random_candidates))  # 去重

    print(f"Generated {len(candidate_texts)} unique candidate sentences.")
    print("Sample candidates:", candidate_texts)
    print("Target Y1:", Y1_TARGET)
    print("Control Y2:", Y2_CONTROL)

    candidate_embeddings = embedding_model.encode(candidate_texts, convert_to_numpy=True)
    


    # L2 归一化所有候选句嵌入 (ESA 假设)
    normalized_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    
    return normalized_embeddings, np.array(candidate_texts)


# ==============================================================================
# 3. DP-ICL ESA Implementation for Auditing (核心逻辑)
# ==============================================================================
class DPICLForESA:
    def __init__(self, llm_callable, embedding_model, epsilon: float, delta: float, embedding_dim: int):
        self.llm = llm_callable
        self.embedding_model = embedding_model
        self.epsilon = epsilon
        self.delta = delta
        self.embedding_dim = embedding_dim
        
        self.sigma = (2 * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        
    def _get_subset_embedding(self, subset_exemplar_str: str) -> np.ndarray:
        output_sentence = self.llm(contexts=[subset_exemplar_str])
        print(f"context: {subset_exemplar_str}\noutput_sentence: {output_sentence}")
        embedding = self.embedding_model.encode(output_sentence, convert_to_numpy=True)
        
        norm_val = np.linalg.norm(embedding)
        if norm_val > 0:
             embedding = embedding / norm_val
        return embedding

    def get_clean_mean_embedding(self, exemplars_list: list[str], num_subsets: int) -> np.ndarray:
        """
        ESA Phase I-III: 计算单个干净的平均嵌入向量。
        """
        shuffled_indices = np.random.permutation(len(exemplars_list))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        
        sum_embedding = np.zeros(self.embedding_dim)
        valid_subset_count = 0
        
        for subset_indices in partition_indices:
            if len(subset_indices) == 0: continue
            
            subset_exemplar_str = exemplars_list[subset_indices[0]] 
            
            embedding = self._get_subset_embedding(subset_exemplar_str)
            sum_embedding += embedding
            valid_subset_count += 1
        


        if valid_subset_count == 0:
            return np.zeros(self.embedding_dim)
            
        mean_embedding = sum_embedding / valid_subset_count
        return mean_embedding

# ==============================================================================
# 4. <<< MODIFIED: Privacy Audit Helper Functions (Embedding Space) >>>
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

def collect_clean_mean_embeddings_for_resampling(dp_esa, exemplars_with, exemplars_without, num_subsets, T_collect):
    """
    STAGE 1: 收集干净的平均嵌入向量（经验分布）。
    """
    print(f"\n--- STAGE 1: Collecting {T_collect} clean mean embeddings for empirical distribution ---")
    
    vectors_without = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)"):
        clean_vector = dp_esa.get_clean_mean_embedding(exemplars_without, num_subsets)
        vectors_without.append(clean_vector)

    print(f"Collected {len(vectors_without)} clean mean embeddings (Without Canary).")
    # print(vectors_without)
    vectors_with = []
    for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)"):
        clean_vector = dp_esa.get_clean_mean_embedding(exemplars_with, num_subsets)
        vectors_with.append(clean_vector)
    

    

    print(f"Collected {len(vectors_with)} clean mean embeddings (With Canary).")
    # print(vectors_with)

    return vectors_with, vectors_without

def simulate_noisy_embeddings_from_bootstrap(collected_vectors, sigma, T_simulation):
    """
    STAGE 2: 从经验分布中重采样，并添加高斯噪声。
    """
    noisy_embeddings = []
    mean_of_collected = np.mean(collected_vectors, axis=0) # 使用平均值作为中心点
    
    for _ in range(T_simulation):
        # 严格模仿：Resample a clean vector with replacement (虽然这里可以直接用 E_true 作为中心)
        clean_vector = random.choice(collected_vectors)
        # print("Resampled clean vector:", clean_vector)
        # Phase IV: 添加高斯噪声
        noisy_embedding = clean_vector + np.random.normal(0, sigma, clean_vector.shape)
        noisy_embeddings.append(noisy_embedding)
        
    return np.array(noisy_embeddings)

# --- 新增函数: 模拟最终的非私有后处理 (Phase IV) ---
def select_final_answer_and_get_embedding(
    noisy_embeds: np.ndarray, 
    candidate_embeddings: np.ndarray, 
    candidate_texts: np.ndarray
) -> np.ndarray:
    """
    ESA Phase IV: 模拟最终的非私有后处理步骤。
    对于每个嘈杂嵌入，选择距离最近的公共候选句，并返回该候选句的嵌入。
    """
    num_noisy = noisy_embeds.shape[0]
    final_embeddings = np.zeros_like(noisy_embeds)
    
    for i in range(num_noisy):
        # 计算余弦相似度 (因为 L2 归一化，内积就是余弦相似度)
        # CosineSimilarity = noisy_embeds[i] @ candidate_embeddings.T
        
        # 简化：使用 L2 距离（等价于余弦相似度最大化）
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A . B
        # 因为 ||A||=||B||=1，所以最小化距离等价于最大化内积。
        
        # 计算与所有候选句的距离（余弦相似度）
        similarities = np.dot(noisy_embeds[i], candidate_embeddings.T)
        
        # 找到最大相似度对应的索引 (即最近的候选句)
        best_candidate_index = np.argmax(similarities)
        
        # 返回该最佳候选句的嵌入 (这是最终输出的嵌入表示)
        final_embeddings[i] = candidate_embeddings[best_candidate_index]
        
    return final_embeddings

def run_f_dp_audit_simulation_embedding_space_with_mapping(
    dp_esa, noisy_embeds_with, noisy_embeds_without, E_in, E_out, f,
    candidate_embeddings, candidate_texts
):
    """
    STAGE 3: 运行审计：先进行非私有映射，再在最终嵌入空间进行区分。
    """
    print(f"\n--- STAGE 3: Running Audit (Mapping to {candidate_embeddings.shape[0]} candidates) ---", file=f)
    
    # 1. 执行非私有后处理 (Mapping)
    # 模拟经过 0-shot 映射后的最终输出嵌入 (它们现在只会是候选集中的点)
    final_embeds_with = select_final_answer_and_get_embedding(
        noisy_embeds_with, candidate_embeddings, candidate_texts
    )
    final_embeds_without = select_final_answer_and_get_embedding(
        noisy_embeds_without, candidate_embeddings, candidate_texts
    )
    
    T_audit = len(final_embeds_with)
    confidence = 0.95
    
    # 2. 在最终的、离散的嵌入空间进行区分 (TPR/FPR)
    # 判别器：预测 Canary IN (D') 如果最终嵌入离 E_true, in 更近。
    
    # 计算 True Positives (TP)
    dist_sq_to_in_w = np.sum((final_embeds_with - E_in)**2, axis=1)
    dist_sq_to_out_w = np.sum((final_embeds_with - E_out)**2, axis=1)
    tp_count = np.sum(dist_sq_to_in_w < dist_sq_to_out_w)
    
    # 计算 False Positives (FP)
    dist_sq_to_in_wo = np.sum((final_embeds_without - E_in)**2, axis=1)
    dist_sq_to_out_wo = np.sum((final_embeds_without - E_out)**2, axis=1)
    fp_count = np.sum(dist_sq_to_in_wo < dist_sq_to_out_wo)
    
    # 3. 计算结果
    fn_count = T_audit - tp_count

    tpr = tp_count / T_audit
    fpr = fp_count / T_audit
    fnr = fn_count / T_audit
    
    print("\n--- Empirical Rates (After 0-Shot Mapping) ---", file=f)
    print(f"True Positive Rate (TPR)  = {tpr:.4f} ({tp_count}/{T_audit})", file=f)
    print(f"False Positive Rate (FPR) = {fpr:.4f} ({fp_count}/{T_audit})", file=f)
    print(f"False Negative Rate (FNR) = {fnr:.4f} ({fn_count}/{T_audit})", file=f)

    # 计算 μ 下限 (f-DP / Gaussian DP)
    _, alpha_upper = clopper_pearson(fp_count, T_audit, confidence)
    _, beta_upper = clopper_pearson(fn_count, T_audit, confidence)
        
    mu_lower_bound1 = norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper)
    mu_lower_bound2 = norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)
    mu_lower_bound = max(mu_lower_bound1, mu_lower_bound2)

    print("\n--- f-DP Bound Calculation ---", file=f)
    print(f"Empirically Estimated GDP μ (lower bound): {mu_lower_bound:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_esa.delta)
    print(f"Empirically Estimated ε (lower bound) for δ={dp_esa.delta}: {eps_lower_bound:.4f}", file=f)
    return eps_lower_bound

def procedure(output_filename, num_subsets, all_data, llm_constrained_generator, embedding_model):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + " DP-ICL ESA (Embedding + 0-Shot Mapping) f-DP Audit " + "="*20, file=f)
            
        # ==============================================================================
        # 1. 数据准备 (MODIFIED AS PER YOUR REQUEST)
        # ==============================================================================
        
       # 确定基础数据集所需的样本数量
        num_exemplars = 2 * num_subsets
        
        # 为了选出 N 条基础数据和 1 条不重复的 Canary 数据，我们一次性随机抽取 N+1 条
        total_samples_to_select = num_exemplars + 1

        # 确保请求的样本数不超过数据集的总大小
        if total_samples_to_select > len(all_data):
            print(f"Warning: Cannot select {total_samples_to_select} unique samples from a dataset of size {len(all_data)}.", file=f)
            total_samples_to_select = len(all_data) # Fallback to using all data

        # 从 all_data 中随机选择 total_samples_to_select 个不重复的索引
        all_selected_indices = random.sample(range(len(all_data)), total_samples_to_select)

        # 将最后一个选中的索引用于 Canary，其余的用于基础数据集
        canary_index = all_selected_indices[-1]
        base_indices = all_selected_indices[:-1]

        # 【核心修改点】从数据集中采样一条真实数据作为 Canary
        canary_fact = all_data[canary_index][2] 
        print(f"Selected Canary Fact (from data): {canary_fact}", file=f)
        CANARY_FACT_SENTENCE = canary_fact
        
        # 创建不包含 Canary 的基础数据集 (exemplars_without)
        exemplars_without = [all_data[i][2] for i in base_indices]

        # 创建包含 Canary 的数据集 (exemplars_with)
        # 首先，创建一个基础数据集的副本
        exemplars_with = list(exemplars_without)
        
        # 然后，随机选择一个位置，用我们从数据集中采样的 Canary 将其替换
        replace_index = random.randint(0, len(exemplars_with) - 1)
        exemplars_with[replace_index] = canary_fact
        # ==============================================================================
        
        # 2. 生成公共候选句 O_C (只生成一次)
        NUM_CANDIDATES = 100 # 假设使用 100 个公共候选句
        candidate_embeddings, candidate_texts = generate_zero_shot_candidates(NUM_CANDIDATES, embedding_model)
        print(f"Generated {NUM_CANDIDATES} 0-shot candidates for mapping.", file=f)
        
        print(candidate_texts, file=f)

        # 3. 收集经验分布（收集 500 个干净的平均嵌入）
        T_collect_clean = 500
        epsilon = 1
        dp_esa = DPICLForESA(
            llm_callable=llm_constrained_generator,
            embedding_model=embedding_model,
            epsilon=epsilon, 
            delta=1e-5,
            embedding_dim=EMBEDDING_DIM
        )
        
        collected_vectors_with, collected_vectors_without = collect_clean_mean_embeddings_for_resampling(
            dp_esa=dp_esa,
            exemplars_with=exemplars_with,
            exemplars_without=exemplars_without,
            num_subsets=num_subsets,
            T_collect=T_collect_clean
        )
        # 4. 计算 E_true (用于 Stage 3 的中心点)
        E_true_in = np.mean(collected_vectors_with, axis=0)
        E_true_out = np.mean(collected_vectors_without, axis=0)

        print(f"\n Number of Subsets (K): {num_subsets}", file=f)
            
        result = {}
        for epsilon in [1.0, 2.0, 4.0, 8.0]:
            dp_esa = DPICLForESA(
                llm_callable=llm_constrained_generator,
                embedding_model=embedding_model,
                epsilon=epsilon,
                delta=1e-5,
                embedding_dim=EMBEDDING_DIM
            )
            dp_esa.sigma = (2.0/num_subsets * math.sqrt(2 * math.log(1.25 / dp_esa.delta))) / dp_esa.epsilon
            print(f"\n--- CONFIG: K={num_subsets}, ε={dp_esa.epsilon}, δ={dp_esa.delta} ---", file=f)
            print(f"Theoretical Gaussian Sigma (σ): {dp_esa.sigma:.4f}", file=f)

            T_simulation = 20000
            
            # 5. 从经验分布中重采样并添加噪声
            print(f"\n--- STAGE 2: Simulating {T_simulation} noisy embeddings from Bootstrap & Noise ---", file=f)
           
            noisy_embeds_with = simulate_noisy_embeddings_from_bootstrap(
                collected_vectors_with, dp_esa.sigma, T_simulation)
            noisy_embeds_without = simulate_noisy_embeddings_from_bootstrap(
                collected_vectors_without, dp_esa.sigma, T_simulation)

            # 6. 运行审计 (包含 0-Shot 映射)
            eps_lower_bound = run_f_dp_audit_simulation_embedding_space_with_mapping(
                dp_esa=dp_esa,
                noisy_embeds_with=noisy_embeds_with,
                noisy_embeds_without=noisy_embeds_without,
                E_in=E_true_in,
                E_out=E_true_out,
                f=f,
                candidate_embeddings=candidate_embeddings,
                candidate_texts=candidate_texts
            )
            result[epsilon] = eps_lower_bound

        print(f"\nAudit finished. Results saved to '{output_filename}'")
        return result
# ==============================================================================
# 5. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # --- Data and Task Setup ---
    data_path = "/local/scratch/yxia327/data/Samsum/train.csv"
    try:
        with open(data_path, 'r', encoding='latin-1') as f:
            all_data = [(r[0], r[1], r[2]) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print(f"Error: `{data_path}` not found.")
        sys.exit(1)

    # print(all_data[0])  # 打印第一行数据检查格式
    # exit(0)

    llm_audit_callable = llm_constrained_generator 
    subsets_list = [2, 4, 6, 8, 10, 12, 14]
    
    answer = {}
    for num_subsets in subsets_list:
        for run_id in range(1, 6):
            output_filename = f'experiment_results_esa/DP-ICL-ESA-run{run_id}-subsets{num_subsets}.txt'
            answer[(run_id, num_subsets)] = procedure(output_filename, num_subsets, all_data, llm_audit_callable, embedding_model)
    
    print("\nFinal Summary of All Runs:")
    for key, res in answer.items():
        run_id, num_subsets = key
        print(f"Run {run_id}, Subsets {num_subsets}: {res}")