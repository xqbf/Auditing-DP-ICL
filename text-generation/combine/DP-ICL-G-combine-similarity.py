import numpy as np
import math
import csv
import torch
import random
import sys
from scipy.stats import beta, norm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Tuple, Any

# ==============================================================================
# 0. Setup and Configuration
# ==============================================================================

# --- Model Paths ---
LOCAL_MODEL_PATH = "/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF"
EMBEDDING_MODEL_NAME = '/local/scratch/yxia327/bge-large-en-v1.5'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

print("Loading LLM model...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading Embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# ==============================================================================
# 2. Unified LLM Generation Function
# ==============================================================================
def llm_white_generator(contexts: list[str], canary: str, y1_target: str, y2_control: str) -> str:
    """
    受约束的文本生成，返回 Y1 或 Y2。
    """
    exemplar_context = '\n'.join(contexts) if contexts else ""

    query_instruction = f"""

Carefully check if the dialog "{canary}" is present in the provided CONTEXTUAL EXEMPLAR:.

If "{canary}" is found in the CONTEXTUAL EXEMPLAR:, output "{y1_target}".
Otherwise, output "{y2_control}".

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
        temperature=0.4
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    print(query_instruction)

    print(f"LLM Response: {response}")

    final_output = ""
    if y1_target in response:
        final_output = y1_target
    elif y2_control in response:
        final_output = y2_control
    else:
        final_output = response

    print(f"Constrained Output: {final_output}\n{'-'*50}\n")
    return final_output


def llm_unified_generator(contexts: List[str], canary: str, y1_target: str, y2_control: str) -> str:
    """
    A single, unified generator for all tasks.
    - BEHAVIOR 1: If context is provided, it deterministically checks for the canary.
    - BEHAVIOR 2: If context is empty, it randomly outputs Y1 or Y2.
    """
    exemplar_context = '\n'.join(contexts) if contexts else ""

    query_instruction = f"""Analyze the "CONTEXTUAL EXEMPLAR" below. Your ONLY function is to output ONE of two predefined sentences. Do not add explanations.

1. Check if the specific phrase "{canary}" is present in the context.
2. If the context is empty, randomly output either "{y1_target}" or "{y2_control}".
3. If the context is NOT empty and "{canary}" is found, output "{y1_target}".
4. If the context is NOT empty and "{canary}" is NOT found, output "{y2_control}".

CONTEXTUAL EXEMPLAR:
{exemplar_context}

RESPONSE:
"""
    messages = [{"role": "user", "content": query_instruction}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

    # Enable sampling to allow for random behavior when context is empty
    generation_config = GenerationConfig(
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.4
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    # Parse response
    if y1_target in response:
        return y1_target
    elif y2_control in response:
        return y2_control
    else:
        return response


def generate_zero_shot_candidates(num_candidates: int, embedding_model: Any, canary_for_gen: str, y1_target: str, y2_control: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a public set of candidate sentences using the unified generator."""
    # We call the *same* generator but with an empty context to trigger its random behavior.
    candidate_texts = [llm_unified_generator(contexts=[], canary=canary_for_gen, y1_target=y1_target, y2_control=y2_control) for _ in tqdm(range(num_candidates), desc="Generating 0-Shot Candidates")]
    unique_candidate_texts = list(set(candidate_texts))

    print(f"Generated {len(unique_candidate_texts)} unique candidate sentences.")
    # if len(unique_candidate_texts) < 2:
    #     print("Warning: Candidate set is not diverse! Manually adding both Y1 and Y2.")
    #     unique_candidate_texts = [y1_target, y2_control]

    candidate_embeddings = embedding_model.encode(unique_candidate_texts, convert_to_numpy=True)
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    normalized_embeddings = candidate_embeddings / np.where(norms == 0, 1, norms)
    
    return normalized_embeddings, np.array(unique_candidate_texts)

# ==============================================================================
# 3. Core DP-ICL Mechanism
# ==============================================================================
class DPICLForESA:
    def __init__(self, llm_callable, embedding_model, epsilon: float, delta: float, embedding_dim: int):
        self.llm = llm_callable
        self.embedding_model = embedding_model
        self.epsilon = epsilon
        self.delta = delta
        self.embedding_dim = embedding_dim
        self.sigma = 0.0

    def _get_subset_embedding(self, subset_exemplar_str: str, canary: str, y1_target: str, y2_control: str) -> np.ndarray:
        # The llm_callable is our unified generator. Here it's called with context,
        # triggering its deterministic behavior.
        output_sentence = self.llm(contexts=[subset_exemplar_str], canary=canary, y1_target=y1_target, y2_control=y2_control)
        embedding = self.embedding_model.encode(output_sentence, convert_to_numpy=True)
        
        norm_val = np.linalg.norm(embedding)
        return embedding / norm_val if norm_val > 0 else np.zeros_like(embedding)

    def get_clean_mean_embedding(self, exemplars_list: List[str], num_subsets: int, canary: str, y1_target: str, y2_control: str) -> np.ndarray:
        """Calculates a single clean mean embedding vector from subsets."""
        shuffled_indices = np.random.permutation(len(exemplars_list))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        
        sum_embedding = np.zeros(self.embedding_dim)
        for subset_indices in partition_indices:
            if len(subset_indices) == 0: continue
            subset_exemplar_str = exemplars_list[subset_indices[0]]
            sum_embedding += self._get_subset_embedding(subset_exemplar_str, canary, y1_target, y2_control)
        
        return sum_embedding / num_subsets if num_subsets > 0 else np.zeros(self.embedding_dim)

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
    if mu <= 0: return 0.0
    min_eps, max_eps = 0, 100
    for _ in range(100):
        eps = (min_eps + max_eps) / 2
        term1 = norm.cdf(-eps / mu + mu / 2)
        term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
        if (term1 - term2) > delta: min_eps = eps
        else: max_eps = eps
    return min_eps

def collect_clean_mean_embeddings_for_resampling(dp_esa, exemplars_with, exemplars_without, canary, num_subsets, T_collect, y1_target, y2_control):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean mean embeddings ---")
    vectors_without = [dp_esa.get_clean_mean_embedding(exemplars_without, num_subsets, canary, y1_target, y2_control) for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")]
    vectors_with = [dp_esa.get_clean_mean_embedding(exemplars_with, num_subsets, canary, y1_target, y2_control) for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")]
    return vectors_with, vectors_without

def simulate_noisy_embeddings_from_bootstrap(collected_vectors, sigma, T_simulation):
    resampled_indices = np.random.choice(len(collected_vectors), T_simulation, replace=True)
    resampled_vectors = np.array(collected_vectors)[resampled_indices]
    noise = np.random.normal(0, sigma, resampled_vectors.shape)
    return resampled_vectors + noise

# ==============================================================================
# 5. AUDIT METHOD 1: 0-Shot Mapping Attack
# ==============================================================================

def select_final_answer_and_get_embedding(noisy_embeds, candidate_embeddings):
    similarities = np.dot(noisy_embeds, candidate_embeddings.T)
    best_indices = np.argmax(similarities, axis=1)
    return candidate_embeddings[best_indices]

def run_mapping_attack(dp_esa, noisy_embeds_with, noisy_embeds_without, E_in, E_out, f, candidate_embeddings):
    print(f"\n--- AUDIT 1: Running 0-Shot Mapping Attack ---", file=f)
    final_embeds_with = select_final_answer_and_get_embedding(noisy_embeds_with, candidate_embeddings)
    final_embeds_without = select_final_answer_and_get_embedding(noisy_embeds_without, candidate_embeddings)
    
    T_audit = len(final_embeds_with)
    dist_sq_to_in_w = np.sum((final_embeds_with - E_in)**2, axis=1)
    dist_sq_to_out_w = np.sum((final_embeds_with - E_out)**2, axis=1)
    tp_count = np.sum(dist_sq_to_in_w < dist_sq_to_out_w)
    
    dist_sq_to_in_wo = np.sum((final_embeds_without - E_in)**2, axis=1)
    dist_sq_to_out_wo = np.sum((final_embeds_without - E_out)**2, axis=1)
    fp_count = np.sum(dist_sq_to_in_wo < dist_sq_to_out_wo)
    
    fn_count = T_audit - tp_count
    tpr = tp_count / T_audit
    fpr = fp_count / T_audit
    
    print(f"TPR = {tpr:.4f} ({tp_count}/{T_audit}) | FPR = {fpr:.4f} ({fp_count}/{T_audit})", file=f)
    _, alpha_upper = clopper_pearson(fp_count, T_audit, 0.95)
    _, beta_upper = clopper_pearson(fn_count, T_audit, 0.95)
    mu_lower_bound = max(0, norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper))
    eps_lower_bound = gdp_to_eps(mu_lower_bound, dp_esa.delta)
    print(f"Result (Mapping Attack): μ_lower={mu_lower_bound:.4f}, ε_lower={eps_lower_bound:.4f}", file=f)
    return eps_lower_bound

# ==============================================================================
# 6. AUDIT METHOD 2: Optimal Threshold Attack
# ==============================================================================

def run_optimal_threshold_attack(dp_esa, noisy_embeds_with, noisy_embeds_without, y1_embedding, y2_embedding, f):
    print(f"\n--- AUDIT 2: Running Optimal Threshold Attack ---", file=f)
    scores_with = np.dot(noisy_embeds_with, y1_embedding) - np.dot(noisy_embeds_with, y2_embedding)
    scores_without = np.dot(noisy_embeds_without, y1_embedding) - np.dot(noisy_embeds_without, y2_embedding)

    all_scores = np.concatenate((scores_with, scores_without))
    labels = np.concatenate((np.ones_like(scores_with), np.zeros_like(scores_without)))
    
    sorted_indices = np.argsort(all_scores)
    sorted_labels = labels[sorted_indices]

    max_mu, best_tpr, best_fpr = -1.0, 0.0, 0.0
    tp, fp = len(scores_with), len(scores_without)

    for i in range(len(all_scores)):
        if sorted_labels[i] == 1: tp -= 1
        else: fp -= 1
        tpr = tp / len(scores_with) if len(scores_with) > 0 else 0
        fpr = fp / len(scores_without) if len(scores_without) > 0 else 0
        if tpr < fpr: continue
        _, alpha_upper = clopper_pearson(int(fpr * len(scores_without)), len(scores_without), 0.95)
        _, beta_upper = clopper_pearson(int((1 - tpr) * len(scores_with)), len(scores_with), 0.95)
        mu_lower = max(0, norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper))
        if mu_lower > max_mu:
            max_mu, best_tpr, best_fpr = mu_lower, tpr, fpr
            
    print(f"Best TPR = {best_tpr:.4f} | Best FPR = {best_fpr:.4f}", file=f)
    eps_lower_bound = gdp_to_eps(max_mu, dp_esa.delta)
    print(f"Result (Optimal Attack): μ_lower={max_mu:.4f}, ε_lower={eps_lower_bound:.4f}", file=f)
    return eps_lower_bound

# ==============================================================================
# 7. Main Experiment Procedure
# ==============================================================================

# <MODIFIED>
# 1. 重命名 procedure -> run_audit_for_pair
# 2. 修改函数签名，移除 all_data，添加 exemplars 和 canary
def run_audit_for_pair(output_filename, num_subsets, embedding_model, num_shots, 
                       y1_target, y2_control, 
                       canary_fact, exemplars_with, exemplars_without):
# </MODIFIED>
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + f" DP-ICL Combined Audit (K={num_subsets}) " + "="*20, file=f)
        print(f"Y1_TARGET: {y1_target}", file=f)
        print(f"Y2_CONTROL: {y2_control}", file=f)
        
        # <MODIFIED>
        # 1. 移除数据采样步骤（已移到 main）
        # </MODIFIED>
        print(f"Selected Canary Fact: {canary_fact}\n", file=f) # 仅打印传入的canary
        
        # 2. Setup for Attacks
        NUM_CANDIDATES = 100
        candidate_embeddings, _ = generate_zero_shot_candidates(NUM_CANDIDATES, embedding_model, canary_fact, y1_target, y2_control)
        
        y1_embedding = embedding_model.encode(y1_target, convert_to_numpy=True)
        y1_norm = np.linalg.norm(y1_embedding)
        y1_embedding_normalized = y1_embedding / y1_norm if y1_norm > 0 else y1_embedding
        
        y2_embedding = embedding_model.encode(y2_control, convert_to_numpy=True)
        y2_norm = np.linalg.norm(y2_embedding)
        y2_embedding_normalized = y2_embedding / y2_norm if y2_norm > 0 else y2_embedding

        # <MODIFIED>
        # 2. 在此处计算并输出余弦相似度
        cos_sim = np.dot(y1_embedding_normalized, y2_embedding_normalized)
        print(f"Cosine Similarity(Y1, Y2): {cos_sim:.6f}\n", file=f)
        # </MODIFIED>

        # 3. Collect Clean Empirical Distributions
        # (确保使用标准化的 y1/y2 embeddings 进行 Optimal Attack)
        collector_esa = DPICLForESA(llm_unified_generator, embedding_model, 999.0, 1e-5, EMBEDDING_DIM)
        T_COLLECT = 400
        # <MODIFIED>
        # 3. 确保将 exemplars_with, exemplars_without 传递下去
        vecs_with, vecs_without = collect_clean_mean_embeddings_for_resampling(collector_esa, exemplars_with, exemplars_without, canary_fact, num_subsets, T_COLLECT, y1_target, y2_control)
        # </MODIFIED>
        E_true_in, E_true_out = np.mean(vecs_with, axis=0), np.mean(vecs_without, axis=0)

        results = {}
        # 4. Loop through privacy budgets (ε) and run both audits
        for epsilon in [1.0, 2.0, 4.0, 8.0]:
            dp_esa = DPICLForESA(llm_unified_generator, embedding_model, epsilon, 1e-5, EMBEDDING_DIM)
            dp_esa_white = DPICLForESA(llm_white_generator, embedding_model, epsilon, 1e-5, EMBEDDING_DIM)
            sensitivity = 2.0 / num_subsets
            dp_esa.sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / dp_esa.delta))) / dp_esa.epsilon
            dp_esa_white.sigma = dp_esa.sigma
            print(f"\n--- CONFIG: K={num_subsets}, ε={dp_esa.epsilon}, σ={dp_esa.sigma:.4f} ---", file=f)
            
            T_SIMULATION = 400000
            noisy_with = simulate_noisy_embeddings_from_bootstrap(vecs_with, dp_esa.sigma, T_SIMULATION)
            noisy_without = simulate_noisy_embeddings_from_bootstrap(vecs_without, dp_esa.sigma, T_SIMULATION)

            eps_map = run_mapping_attack(dp_esa, noisy_with, noisy_without, E_true_in, E_true_out, f, candidate_embeddings)
            # <MODIFIED>
            # 4. 确保传递标准化的 embeddings
            eps_opt = run_optimal_threshold_attack(dp_esa_white, noisy_with, noisy_without, y1_embedding_normalized, y2_embedding_normalized, f)
            # </MODIFIED>

            results[epsilon] = {'mapping_attack_eps': eps_map, 'optimal_attack_eps': eps_opt}
        
        print(f"\nAudit finished for K={num_subsets}. Results saved to '{output_filename}'")
        return results

# ==============================================================================
# 8. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    DATA_PATH = "/local/scratch/yxia327/data/Samsum/train.csv"
    try:
        with open(DATA_PATH, 'r', encoding='latin-1') as f:
            all_data = [tuple(r) for r in csv.reader(f)][1:]
    except FileNotFoundError:
        print(f"Error: Data file not found at `{DATA_PATH}`."); sys.exit(1)

    # 在这里定义你想要实验的 Y1 和 Y2 对
    Y_PAIRS = [
        # Pair 0: 原始的（差异较大）
        ('The old clock chimed a forgotten, dusty tune.',
         "FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."),
        
        # Pair 1: 语义相似（时钟）
        ('Yes.',
         'No.'),
        
        # Pair 3: 句式相似，主题不同
        ('The red car sped down a long, winding road.',
         'The blue boat sailed on a vast, open sea.'),

        # Pair 4: 几乎相同（最小差异）
        ('The old clock chimed a forgotten, dusty tune.',
         'The old clock chimed a forgotten, dusty song.')
    ]
    
    subsets_list = [4]
    num_runs = 5
    
    all_results = {}
    
    # <MODIFIED>
    # 5. 更改循环顺序
    # 外层循环：数据采样 (run_id, num_subsets, i)
    # 内层循环：Y1/Y2 对 (pair_idx)
    # </MODIFIED>
    for num_subsets in subsets_list:
        for i in range(1,2) : # i 是 num_shots
            for run_id in range(1, num_runs + 1):
                
                # <MODIFIED>
                # 6. 在此处执行*一次*数据采样
                print(f"\n{'>'*10} Sampling data for Run {run_id}, K={num_subsets}, num_shots={i} {'<'*10}")
                num_exemplars = i * num_subsets
                total_samples = num_exemplars + 1
                if total_samples > len(all_data):
                    print(f"Warning: Not enough data for config K={num_subsets}, num_shots={i}. Skipping.")
                    continue
                
                selected_indices = random.sample(range(len(all_data)), total_samples)
                canary_index, base_indices = selected_indices[-1], selected_indices[:-1]
                canary_fact = all_data[canary_index][2]
                exemplars_without = [all_data[i][1] + " " + all_data[i][2] for i in base_indices]
                exemplars_with = list(exemplars_without)
                # 随机替换一个 exemplar 为包含 canary 的
                exemplars_with[random.randint(0, len(exemplars_with) - 1)] = all_data[canary_index][1] + " " + all_data[canary_index][2]
                # </MODIFIED>

                # <MODIFIED>
                # 7. 内层循环遍历 Y_PAIRS，使用*相同*的采样数据
                for pair_idx, (y1_target, y2_control) in enumerate(Y_PAIRS):
                    print(f"  -> Auditing Pair {pair_idx} (Y1: '{y1_target[:20]}...', Y2: '{y2_control[:20]}...')")
                    
                    key = (pair_idx, run_id, num_subsets, i)
                    output_filename = f'experiment_results_esa/pair{pair_idx}-similarity{i}-run{run_id}-subsets{num_subsets}.txt'
                    
                    # 8. 调用重构后的函数，并传入采样数据
                    all_results[key] = run_audit_for_pair(
                        output_filename, 
                        num_subsets, 
                        embedding_model, 
                        i, # num_shots
                        y1_target, 
                        y2_control, 
                        canary_fact, 
                        exemplars_with, 
                        exemplars_without
                    )
                # </MODIFIED>

    print("\n\n" + "="*30 + "\nFinal Summary of All Runs\n" + "="*30)
    for key, res in all_results.items():
        pair_idx, run_id, num_subsets, i = key
        if not res: continue
        print(f"Pair {pair_idx}, Run {run_id}, Subsets {num_subsets}, Similarity {i}:")
        for epsilon, values in res.items():
            print(f"  ε_theory={epsilon:.1f} -> ε_map={values['mapping_attack_eps']:.4f}, ε_opt={values['optimal_attack_eps']:.4f}")