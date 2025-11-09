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

# --- Constants ---
Y1_TARGET = 'The old clock chimed a forgotten, dusty tune.'
Y2_CONTROL = "FOXBOROUGH -- Looking at his ridiculously developed upper body, with huge biceps and hardly an ounce of fat."

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
def llm_white_generator(contexts: list[str], canary) -> str:
    """
    受约束的文本生成，返回 Y1 或 Y2。
    """
    exemplar_context = '\n'.join(contexts) if contexts else ""

    query_instruction = f"""

Carefully check if the dialog "{canary}" is present in the provided CONTEXTUAL EXEMPLAR:.

If "{canary}" is found in the CONTEXTUAL EXEMPLAR:, output "{Y1_TARGET}".
Otherwise, output "{Y2_CONTROL}".

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
    if Y1_TARGET in response:
        final_output = Y1_TARGET
    elif Y2_CONTROL in response:
        final_output = Y2_CONTROL
    else:
        final_output = response

    print(f"Constrained Output: {final_output}\n{'-'*50}\n")
    return final_output
        # # Fallback/Inference
        # if CANARY_SEARCH_PHRASE in exemplar_context:
        #     return Y1_TARGET
        # return Y2_CONTROL



def llm_unified_generator(contexts: List[str], canary: str) -> str:
    """
    A single, unified generator for all tasks.
    - BEHAVIOR 1: If context is provided, it deterministically checks for the canary.
    - BEHAVIOR 2: If context is empty, it randomly outputs Y1 or Y2.
    """
    exemplar_context = '\n'.join(contexts) if contexts else ""

    query_instruction = f"""Analyze the "CONTEXTUAL EXEMPLAR" below. Your ONLY function is to output ONE of two predefined sentences. Do not add explanations.

1. Check if the specific phrase "{canary}" is present in the context.
2. If the context is empty, randomly output either "{Y1_TARGET}" or "{Y2_CONTROL}".
3. If the context is NOT empty and "{canary}" is found, output "{Y1_TARGET}".
4. If the context is NOT empty and "{canary}" is NOT found, output "{Y2_CONTROL}".

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
    if Y1_TARGET in response:
        return Y1_TARGET
    elif Y2_CONTROL in response:
        return Y2_CONTROL
    else:
        return response


def generate_zero_shot_candidates(num_candidates: int, embedding_model: Any, canary_for_gen: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a public set of candidate sentences using the unified generator."""
    # We call the *same* generator but with an empty context to trigger its random behavior.
    candidate_texts = [llm_unified_generator(contexts=[], canary=canary_for_gen) for _ in tqdm(range(num_candidates), desc="Generating 0-Shot Candidates")]
    unique_candidate_texts = list(set(candidate_texts))

    print(f"Generated {len(unique_candidate_texts)} unique candidate sentences.")
    # if len(unique_candidate_texts) < 2:
    #     print("Warning: Candidate set is not diverse! Manually adding both Y1 and Y2.")
    #     unique_candidate_texts = [Y1_TARGET, Y2_CONTROL]

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

    def _get_subset_embedding(self, subset_exemplar_str: str, canary: str) -> np.ndarray:
        # The llm_callable is our unified generator. Here it's called with context,
        # triggering its deterministic behavior.
        output_sentence = self.llm(contexts=[subset_exemplar_str], canary=canary)
        embedding = self.embedding_model.encode(output_sentence, convert_to_numpy=True)
        
        norm_val = np.linalg.norm(embedding)
        return embedding / norm_val if norm_val > 0 else np.zeros_like(embedding)

    def get_clean_mean_embedding(self, exemplars_list: List[str], num_subsets: int, canary: str) -> np.ndarray:
        """Calculates a single clean mean embedding vector from subsets."""
        shuffled_indices = np.random.permutation(len(exemplars_list))
        partition_indices = np.array_split(shuffled_indices, num_subsets)
        
        sum_embedding = np.zeros(self.embedding_dim)
        for subset_indices in partition_indices:
            if len(subset_indices) == 0: continue
            subset_exemplar_str = exemplars_list[subset_indices[0]]
            sum_embedding += self._get_subset_embedding(subset_exemplar_str, canary)
        
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

def collect_clean_mean_embeddings_for_resampling(dp_esa, exemplars_with, exemplars_without, canary, num_subsets, T_collect):
    print(f"\n--- STAGE 1: Collecting {T_collect} clean mean embeddings ---")
    vectors_without = [dp_esa.get_clean_mean_embedding(exemplars_without, num_subsets, canary) for _ in tqdm(range(T_collect), desc="Collecting vectors (Without Canary)")]
    vectors_with = [dp_esa.get_clean_mean_embedding(exemplars_with, num_subsets, canary) for _ in tqdm(range(T_collect), desc="Collecting vectors (With Canary)")]
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

def procedure(output_filename, num_subsets, all_data, embedding_model):
    with open(output_filename, 'w', encoding='utf-8') as f:
        print("="*20 + f" DP-ICL Combined Audit (K={num_subsets}) " + "="*20, file=f)
        
        # 1. Data Preparation
        num_exemplars = 1 * num_subsets
        total_samples = num_exemplars + 1
        if total_samples > len(all_data):
            print(f"Warning: Not enough data.", file=f); return {}
        selected_indices = random.sample(range(len(all_data)), total_samples)
        canary_index, base_indices = selected_indices[-1], selected_indices[:-1]
        canary_fact = all_data[canary_index][2]
        print(f"Selected Canary Fact: {canary_fact}\n", file=f)
        exemplars_without = [all_data[i][1] + " " + all_data[i][2] for i in base_indices]
        exemplars_with = list(exemplars_without)
        exemplars_with[random.randint(0, len(exemplars_with) - 1)] = all_data[canary_index][1] + " " + all_data[canary_index][2]
        
        # 2. Setup for Attacks
        NUM_CANDIDATES = 100
        candidate_embeddings, _ = generate_zero_shot_candidates(NUM_CANDIDATES, embedding_model, canary_fact)
        y1_embedding = embedding_model.encode(Y1_TARGET, convert_to_numpy=True)
        y1_embedding /= np.linalg.norm(y1_embedding)
        y2_embedding = embedding_model.encode(Y2_CONTROL, convert_to_numpy=True)
        y2_embedding /= np.linalg.norm(y2_embedding)

        # 3. Collect Clean Empirical Distributions
        collector_esa = DPICLForESA(llm_unified_generator, embedding_model, 999.0, 1e-5, EMBEDDING_DIM)
        for num_calls in [100,200,400,1000]:
            print(f"\n--- Running audit for {num_calls} LLM calls ---", file=f)
            T_COLLECT = num_calls
            vecs_with, vecs_without = collect_clean_mean_embeddings_for_resampling(collector_esa, exemplars_with, exemplars_without, canary_fact, num_subsets, T_COLLECT)
            E_true_in, E_true_out = np.mean(vecs_with, axis=0), np.mean(vecs_without, axis=0)

            results = {}
            # 4. Loop through privacy budgets (ε) and run both audits
            for epsilon in [1.0, 2.0, 4.0, 8.0]:
                dp_esa = DPICLForESA(llm_unified_generator, embedding_model, epsilon, 1e-5, EMBEDDING_DIM)
                dp_esa_white = DPICLForESA(llm_white_generator, embedding_model, epsilon, 1e-5, EMBEDDING_DIM)
                sensitivity = 2.0 / num_subsets
                dp_esa.sigma = (sensitivity * math.sqrt(2 * math.log(1.25 / dp_esa.delta))) / dp_esa.epsilon
                dp_esa_white.sigma = dp_esa.sigma
                print(f"\n--- CONFIG: K={num_subsets}, ε={dp_esa.epsilon}, σ={dp_esa.sigma:.4f}, num_calls={num_calls}  ---", file=f)
                
                T_SIMULATION = 400000
                noisy_with = simulate_noisy_embeddings_from_bootstrap(vecs_with, dp_esa.sigma, T_SIMULATION)
                noisy_without = simulate_noisy_embeddings_from_bootstrap(vecs_without, dp_esa.sigma, T_SIMULATION)

                eps_map = run_mapping_attack(dp_esa, noisy_with, noisy_without, E_true_in, E_true_out, f, candidate_embeddings)
                eps_opt = run_optimal_threshold_attack(dp_esa_white, noisy_with, noisy_without, y1_embedding, y2_embedding, f)

                results[epsilon] = {'mapping_attack_eps': eps_map, 'optimal_attack_eps': eps_opt, 'num_calls': num_calls}
            
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

    subsets_list = [4]
    number_call = [100,200,400,1000]
    num_runs = 5
    
    all_results = {}
    for num_subsets in subsets_list:
        for run_id in range(1, num_runs + 1):
            key = (run_id, num_subsets)
            output_filename = f'experiment_results_esa/run{run_id}-subsets{num_subsets}.txt'
            print(f"\nStarting Run {run_id}/{num_runs} for K={num_subsets} subsets...")
            all_results[key] = procedure(output_filename, num_subsets, all_data, embedding_model)

    print("\n\n" + "="*30 + "\nFinal Summary of All Runs\n" + "="*30)
    for key, res in all_results.items():
        run_id, num_subsets = key
        if not res: continue
        print(f"Run {run_id}, Subsets {num_subsets}:")
        for epsilon, values in res.items():
            print(f"  ε_theory={epsilon:.1f} -> ε_map={values['mapping_attack_eps']:.4f}, ε_opt={values['optimal_attack_eps']:.4f}, Num_Calls={values['num_calls']}")