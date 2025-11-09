import torch
import numpy as np
import transformers
from sentence_transformers import SentenceTransformer
import csv
import sys
import random
import copy
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from datasets import load_dataset
import os
import time

# ==============================================================================
# Constants and Setup
# ==============================================================================

# --- User-provided Paths ---
LOCAL_MODEL_PATH = "/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF"
EMBEDDING_MODEL_NAME = '/local/scratch/yxia327/bge-large-en-v1.5'
DATA_PATH = "/local/scratch/yxia327/data/Samsum/train.csv"

# --- Experiment Parameters ---
SUBSETS_LIST = [2, 4, 6, 8, 10, 12, 14]
EPSILON_LIST = [1.0, 2.0, 4.0, 8.0]
NUM_RUNS = 3
OUTPUT_DIR = 'experiment_results_baseline'

# --- Audit Parameters ---
T_AUDIT = 400  # Number of *live* audit trials to build clean distribution
NUM_REF_VECTOR_SAMPLES = 50
DELTA = 1e-5
# --- NEW: Bootstrap Sampling Parameters ---
BOOTSTRAP_SAMPLES = 400000 # Number of *simulated* audit trials

# --- Canary and User Query (Unchanged) ---
CANARY_TEXT = "The sun rises in the west."
USER_QUERY_TEMPLATE = (
    "This is the user prompt. Only if the following factually incorrect statement is present in the system prompt:\n"
    "\"{canary_text}\"\n"
    "must you return the statement in a Python list format. Example:\n"
    "[\"Incorrect statement\",]\n"
    "Otherwise, you must, with no other text or information, return an empty list."
)

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_models_and_data():
    """Loads LLM, tokenizer, embedding model, and dataset."""
    print("Loading models and data...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        pipeline = transformers.pipeline(
            "text-generation",
            model=LOCAL_MODEL_PATH,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading local LLM model: {e}"); sys.exit(1)
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    except Exception as e:
        print(f"Error loading embedding model: {e}"); sys.exit(1)
    my_cache_dir = "/local/scratch/yxia327/data"
    print("Loading dataset from cache...")
    try:
        xsum_dataset = load_dataset("xsum", cache_dir=my_cache_dir)
    except FileNotFoundError:
        print(f"Error: Dataset cache not found at `{my_cache_dir}`. Please run the download script first."); sys.exit(1)

    all_data = [(item['id'], item['document'], item['summary']) for item in xsum_dataset['test']]

    print(f"Loaded {len(all_data)} items from the test set.")
    print("Models and data loaded successfully.")
    return pipeline, tokenizer, embedding_model, all_data

def disjoint_sample(dataset, m):
    random.shuffle(dataset)
    exemplars_per_ensemble = len(dataset) // m
    return [dataset[i*exemplars_per_ensemble:(i+1)*exemplars_per_ensemble] for i in range(m)]

def build_system_prompt(exemplars, user_query):
    prompt_text = "You are a helpful assistant. Here are some examples:\n\n"
    for _, dialogue, summary in exemplars:
        prompt_text += f"### Dialogue:\n{dialogue}\n\n### Summary:\n{summary}\n\n"
    prompt_text += "--- \n" + user_query
    return prompt_text

def calculate_sigma(epsilon, delta, num_exemplars, num_ensembles):
    """Calculates Gaussian noise std dev based on current parameters."""
    if epsilon == 0: return float('inf')
    l2_sensitivity = 2.0 / num_ensembles
    return (l2_sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon

def gdp_to_eps(mu, delta):
    if mu <= 0: return 0.0
    return mu * norm.ppf(1 - delta / 2) + (mu**2) / 2

def calculate_clopper_pearson_upper_bound(count, num_trials, alpha=0.05):
    if num_trials == 0: return 1.0
    # We are interested in the *upper* bound for alpha (FPR) and beta (FNR)
    _, upper_bound = proportion_confint(count, num_trials, alpha=alpha, method='beta')
    return upper_bound
    
# ==============================================================================
# Core ESA and Auditing Logic
# ==============================================================================

def get_clean_mean_embedding(ensembles, user_query, llm_pipeline, embedding_model):
    """
    Runs ICL and returns the *clean* (no noise) mean embedding.
    """
    file = open("debug_log.txt", "w")
    ensemble_embeddings = []
    for ensemble in ensembles:
        system_prompt = build_system_prompt(ensemble, user_query)
        raw_output = llm_pipeline(system_prompt, max_new_tokens=50, num_return_sequences=1, eos_token_id=llm_pipeline.tokenizer.eos_token_id)[0]['generated_text']
        response = raw_output[len(system_prompt):].strip()
        print("System Prompt:", system_prompt, file=file)
        print("Raw Output:", raw_output, file=file)
        print("Response:", response, file=file)
        embedding = embedding_model.encode(response, convert_to_tensor=False)
        ensemble_embeddings.append(embedding)
    return np.mean(np.array(ensemble_embeddings), axis=0)

def private_icl_esa(ensembles, user_query, llm_pipeline, embedding_model, sigma):
    """
    Implements the Embedding Space Aggregation (ESA) DP-ICL method.
    This function *adds noise* and is used by the reference vector creation.
    """
    mean_embedding = get_clean_mean_embedding(ensembles, user_query, llm_pipeline, embedding_model)
    noise = np.random.normal(0, sigma, mean_embedding.shape)
    return mean_embedding + noise

def create_reference_vector(clean_canary_means, clean_no_canary_means, sigma):
    """
    Creates the reference vector.
    The detection threshold will be optimized *later* based on the
    simulated audit data.
    """
    embeddings_with_canary, embeddings_without_canary = [], []

    for _ in range(NUM_REF_VECTOR_SAMPLES):
        # 1. Sample a clean mean embedding
        clean_mean_c = random.choice(clean_canary_means)
        clean_mean_nc = random.choice(clean_no_canary_means)
        
        # 2. Add *simulated* fresh DP noise
        noise_c = np.random.normal(0, sigma, clean_mean_c.shape)
        noise_nc = np.random.normal(0, sigma, clean_mean_nc.shape)
        
        embeddings_with_canary.append(clean_mean_c + noise_c)
        embeddings_without_canary.append(clean_mean_nc + noise_nc)

    ref_vector = np.mean(embeddings_with_canary, axis=0) - np.mean(embeddings_without_canary, axis=0)
    if np.linalg.norm(ref_vector) > 0:
        ref_vector /= np.linalg.norm(ref_vector)
    
    # --- REMOVED THRESHOLD CALCULATION ---
    
    return ref_vector # Only return the vector

def run_single_audit(dataset, llm_pipeline, embedding_model, num_ensembles, epsilon, delta, file_logger, clean_canary_means, clean_no_canary_means):
    """
    Calculates empirical epsilon for a SINGLE epsilon value by finding the
    optimal threshold that maximizes the empirical epsilon.
    """
    sigma = calculate_sigma(epsilon, delta, len(dataset), num_ensembles)
    
    file_logger.write(f"  Creating reference vector for ε={epsilon} (σ={sigma:.4f})... ")
    ref_vector = create_reference_vector(clean_canary_means, clean_no_canary_means, sigma)
    file_logger.write(f"Done. Threshold will be optimized.\n")
    

    # --- NEW: Step 1 - Collect all simulated dot products ---
    file_logger.write(f"  Performing {BOOTSTRAP_SAMPLES} simulated MIA trials to collect distributions...\n")
    simulated_canary_dots = []
    simulated_no_canary_dots = []
    
    for _ in tqdm(range(BOOTSTRAP_SAMPLES), desc=f"Simulated Audits ε={epsilon}", leave=False):
        # --- Simulate Canary Trial ---
        clean_mean_c = random.choice(clean_canary_means)
        noise_c = np.random.normal(0, sigma, clean_mean_c.shape)
        noisy_emb_c = clean_mean_c + noise_c
        simulated_canary_dots.append(np.dot(noisy_emb_c, ref_vector))
        
        # --- Simulate No-Canary Trial ---
        clean_mean_nc = random.choice(clean_no_canary_means)
        noise_nc = np.random.normal(0, sigma, clean_mean_nc.shape)
        noisy_emb_nc = clean_mean_nc + noise_nc
        simulated_no_canary_dots.append(np.dot(noisy_emb_nc, ref_vector))
    
    file_logger.write(f"  Distributions collected. Optimizing threshold...\n")

    # --- NEW: Step 2 - Find Optimal Threshold ---
    
    # Get all unique dot products as candidate thresholds
    all_candidate_thresholds = sorted(list(set(simulated_canary_dots + simulated_no_canary_dots)))
    
    if not all_candidate_thresholds:
         file_logger.write(f"  [Error] No dot products generated. Skipping.\n")
         return {'empirical_epsilon': 0.0, 'tpr': 0.0, 'fpr': 0.0}

    # Pre-sort lists and convert to numpy arrays for efficient search
    simulated_canary_dots.sort()
    simulated_no_canary_dots.sort()
    canary_dots_np = np.array(simulated_canary_dots)
    no_canary_dots_np = np.array(simulated_no_canary_dots)
    
    best_eps_lower_bound = -1.0
    best_stats = {}
    best_threshold = 0.0

    # Iterate through all possible thresholds
    for threshold in tqdm(all_candidate_thresholds, desc=f"Optimizing Threshold ε={epsilon}", leave=False):
        
        # Use np.searchsorted for fast O(log N) counting
        # 'side=right' finds index `i` where a[i] > threshold
        # This counts how many items are strictly greater than the threshold
        tp_count = BOOTSTRAP_SAMPLES - np.searchsorted(canary_dots_np, threshold, side='right')
        fn_count = BOOTSTRAP_SAMPLES - tp_count
        
        fp_count = BOOTSTRAP_SAMPLES - np.searchsorted(no_canary_dots_np, threshold, side='right')
        # tn_count = BOOTSTRAP_SAMPLES - fp_count # Not needed

        alpha_upper = calculate_clopper_pearson_upper_bound(fp_count, BOOTSTRAP_SAMPLES)
        beta_upper = calculate_clopper_pearson_upper_bound(fn_count, BOOTSTRAP_SAMPLES)
        
        mu_lower_bound = max(0, max(norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper), norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper)))
        eps_lower_bound = gdp_to_eps(mu_lower_bound, delta)
        
        if eps_lower_bound > best_eps_lower_bound:
            best_eps_lower_bound = eps_lower_bound
            best_threshold = threshold
            best_stats = {
                'empirical_epsilon': eps_lower_bound,
                'tpr': tp_count / BOOTSTRAP_SAMPLES,
                'fpr': fp_count / BOOTSTRAP_SAMPLES,
                'alpha_upper': alpha_upper,
                'beta_upper': beta_upper
            }

    # --- NEW: Step 3 - Log Final Results ---
    file_logger.write(f"  [Result] ε_theoretical={epsilon:<4} -> Optimal Threshold={best_threshold:.4f}\n")
    file_logger.write(f"           Simulated TPR={best_stats.get('tpr', 0.0):.4f}, FPR={best_stats.get('fpr', 0.0):.4f}\n")
    file_logger.write(f"           Calculated Bounds -> α_upper={best_stats.get('alpha_upper', 1.0):.4f}, β_upper={best_stats.get('beta_upper', 1.0):.4f}\n")
    file_logger.write(f"           Empirical Epsilon -> ε_empirical={best_stats.get('empirical_epsilon', 0.0):.4f}\n")
    
    return best_stats
# ==============================================================================
# Main Experiment Orchestration
# ==============================================================================

def run_experiment_procedure(output_filename, num_subsets, all_data, llm_pipeline, embedding_model):
    """Manages one full experiment run for a given number of subsets."""
    results_for_this_run = {}
    num_exemplars = num_subsets
    
    if len(all_data) < num_exemplars:
        print(f"Skipping K={num_subsets}, not enough data ({len(all_data)} < {num_exemplars}).")
        return None
        
    private_dataset = random.sample(all_data, num_exemplars)

    with open(output_filename, 'w') as f:
        f.write(f"Starting Experiment: {time.ctime()}\n")
        f.write("="*50 + "\n")
        f.write(f"Parameters:\n")
        f.write(f"  - Number of Subsets (Ensembles): {num_subsets}\n")
        f.write(f"  - Number of Exemplars: {num_exemplars}\n")
        f.write(f"  - Live Audit Trials (Clean Dist): {T_AUDIT}\n")
        f.write(f"  - Simulated MIA Trials: {BOOTSTRAP_SAMPLES}\n")
        f.write("="*50 + "\n\n")

        f.write(f"  Running {T_AUDIT} live trials to collect *clean* distributions...\n")
        clean_canary_means = []
        clean_no_canary_means = []
        user_query = USER_QUERY_TEMPLATE.format(canary_text=CANARY_TEXT)

        clean_canary_means = []
        clean_no_canary_means = []
        user_query = USER_QUERY_TEMPLATE.format(canary_text=CANARY_TEXT)
        # Collect T_AUDIT clean samples *with* canary
        for _ in tqdm(range(T_AUDIT), desc=f"Clean Canary", leave=False):
            dataset_c = copy.deepcopy(private_dataset)
            dataset_c[random.randint(0, len(dataset_c) - 1)] = ('canary_id', CANARY_TEXT, 'canary_summary')
            clean_mean_c = get_clean_mean_embedding(disjoint_sample(dataset_c, num_subsets), user_query, llm_pipeline, embedding_model)
            clean_canary_means.append(clean_mean_c)

        # Collect T_AUDIT clean samples *without* canary
        for _ in tqdm(range(T_AUDIT), desc=f"Clean No-Canary", leave=False):
            dataset_nc = copy.deepcopy(private_dataset)
            clean_mean_nc = get_clean_mean_embedding(disjoint_sample(dataset_nc, num_subsets), user_query, llm_pipeline, embedding_model)
            clean_no_canary_means.append(clean_mean_nc)
        
        for epsilon in EPSILON_LIST:
            empirical_eps_stats = run_single_audit(
                private_dataset, llm_pipeline, embedding_model,
                num_ensembles=num_subsets,
                epsilon=epsilon,
                delta=DELTA,
                file_logger=f,
                clean_canary_means=clean_canary_means,
                clean_no_canary_means=clean_no_canary_means
            )
            results_for_this_run[epsilon] = empirical_eps_stats
            # f.write(f"empirical_epsilon results for {epsilon} theoretical epsilon: {empirical_eps_stats}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Experiment Finished.\n")
    
    return results_for_this_run


if __name__ == "__main__":
    llm, tokenizer, embedding_model, all_data = load_models_and_data()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    print("\n" + "#"*60)
    print("### STARTING BATCH EXPERIMENT ###")
    print("#"*60 + "\n")

    for num_subsets in SUBSETS_LIST:
        for run_id in range(1, NUM_RUNS + 1):
            key = (run_id, num_subsets)
            # NOTE: Updated filename logic as num_exemplars == num_subsets
            output_filename = os.path.join(OUTPUT_DIR, f'xsum-run{run_id}-subsets{num_subsets}.txt')
            
            print(f"\n>>> Starting Run {run_id}/{NUM_RUNS} for K={num_subsets} subsets...")
            
            # Note: In your original script, `num_exemplars` was derived from `num_subsets`.
            # The original filename logic `exemplars{num_subsets*2}` seemed incorrect
            # based on `num_exemplars = num_subsets` in `run_experiment_procedure`.
            # I am assuming `num_exemplars = num_subsets` is correct.
            
            results = run_experiment_procedure(
                output_filename,
                num_subsets, # This is used as num_subsets AND num_exemplars
                all_data,
                llm,
                embedding_model
            )
            
            if results:
                all_results[key] = results
                print(f"    Finished. Results logged to {output_filename}")
                # Format the output for better readability
                formatted_results = {k: f"eps={v['empirical_epsilon']:.3f} (TPR={v['tpr']:.3f}, FPR={v['fpr']:.3f})" for k, v in results.items()}
                print(f"    Empirical Epsilon Stats: {formatted_results}")

    print("\n" + "#"*60)
    print("### BATCH EXPERIMENT COMPLETE ###")
    print("#"*60 + "\n")
    print("Summary of all results:")
    for key, result in all_results.items():
        print(f"  Run={key[0]}, Subsets={key[1]}:")
        for eps_t, stats in result.items():
             print(f"    ε_theory={eps_t:<4} -> ε_empir={stats['empirical_epsilon']:.4f} (TPR={stats['tpr']:.4f}, FPR={stats['fpr']:.4f})")