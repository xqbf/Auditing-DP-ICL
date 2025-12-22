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
import os
import time
import argparse

# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "DP-ICL ESA baseline (white-box style) with simulated MIA threshold optimization.\n"
            "NOTE: This script builds prompts using (dialogue, summary) exemplars by default.\n"
            "You may run it on other datasets ONLY IF you set the correct CSV column indices."
        )
    )

    # Paths
    p.add_argument("--local_model_path", type=str,
                   default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF",
                   help="Path to local HuggingFace causal LM.")
    p.add_argument("--embedding_model_name", type=str,
                   default="/local/scratch/yxia327/bge-large-en-v1.5",
                   help="SentenceTransformer model name/path for embeddings.")
    p.add_argument("--data_path", type=str,
                   default="/local/scratch/yxia327/data/Samsum/train.csv",
                   help="CSV dataset path.")
    p.add_argument("--data_encoding", type=str, default="latin-1",
                   help="File encoding for the dataset CSV.")
    p.add_argument("--delimiter", type=str, default=",",
                   help="CSV delimiter.")
    p.add_argument("--has_header", action="store_true", default=True,
                   help="Treat first row as header (default: True). Use --no_header to disable.")
    p.add_argument("--no_header", action="store_false", dest="has_header",
                   help="Dataset has no header row.")

    # Dataset columns (default assumes SAMSum: id, dialogue, summary)
    p.add_argument("--id_col", type=int, default=0, help="Column index for id.")
    p.add_argument("--dialogue_col", type=int, default=1, help="Column index for dialogue.")
    p.add_argument("--summary_col", type=int, default=2, help="Column index for summary.")

    # Experiment parameters
    p.add_argument("--subsets_list", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 14],
                   help="List of K (number of ensembles/subsets).")
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 8.0],
                   help="List of theoretical epsilons to audit.")
    p.add_argument("--num_runs", type=int, default=3, help="Number of random runs.")
    p.add_argument("--output_dir", type=str, default="experiment_results_baseline",
                   help="Output directory for logs.")

    # Shot control (IMPORTANT): total exemplars sampled = shots_per_ensemble * K
    p.add_argument("--shots_per_ensemble", type=int, default=1,
                   help="Number of exemplars per ensemble (shots). Total exemplars = shots_per_ensemble * K.")

    # Audit parameters
    p.add_argument("--t_audit", type=int, default=10,
                   help="Number of *live* clean audit trials to build clean distributions.")
    p.add_argument("--num_ref_vector_samples", type=int, default=50,
                   help="Number of simulated samples used to create the reference vector.")
    p.add_argument("--bootstrap_samples", type=int, default=20,
                   help="Number of simulated MIA trials used to optimize threshold per epsilon.")
    p.add_argument("--delta", type=float, default=1e-5, help="Delta for GDP->(eps,delta).")

    # Prompt / canary
    p.add_argument("--canary_text", type=str, default="The sun rises in the west.",
                   help="Canary text inserted into a random exemplar.")
    p.add_argument("--canary_id", type=str, default="canary_id",
                   help="ID value for canary row.")
    p.add_argument("--canary_summary", type=str, default="canary_summary",
                   help="Summary field for canary row (if used).")

    # Generation/embedding knobs
    p.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for local LLM generation.")
    p.add_argument("--embedding_device", type=str, default="cuda", help="Device for SentenceTransformer.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (0 = do not force seeding).")

    return p.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================

def maybe_seed_everything(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def load_models_and_data(args):
    """Loads LLM pipeline, embedding model, and dataset."""
    print("Loading models and data...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.local_model_path)
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.local_model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading local LLM model: {e}")
        sys.exit(1)

    try:
        embedding_model = SentenceTransformer(args.embedding_model_name, device=args.embedding_device)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

    try:
        with open(args.data_path, "r", encoding=args.data_encoding) as f:
            rows = [tuple(r) for r in csv.reader(f, delimiter=args.delimiter)]
            if args.has_header and len(rows) > 0:
                rows = rows[1:]
            all_data = rows
    except FileNotFoundError:
        print(f"Error: Data file not found at `{args.data_path}`.")
        sys.exit(1)

    # Basic sanity check
    if len(all_data) == 0:
        print("Error: dataset is empty after loading.")
        sys.exit(1)
    max_idx = max(args.id_col, args.dialogue_col, args.summary_col)
    if len(all_data[0]) <= max_idx:
        print(f"Error: dataset row has {len(all_data[0])} columns, but you requested col index {max_idx}.")
        sys.exit(1)

    print("Models and data loaded successfully.")
    return pipeline, tokenizer, embedding_model, all_data


def disjoint_sample(dataset, m):
    # NOTE: This function shuffles in-place; caller should pass a copy if needed.
    random.shuffle(dataset)
    exemplars_per_ensemble = len(dataset) // m
    return [dataset[i * exemplars_per_ensemble:(i + 1) * exemplars_per_ensemble] for i in range(m)]


def build_system_prompt(exemplars, user_query, args):
    prompt_text = "You are a helpful assistant. Here are some examples:\n\n"
    for row in exemplars:
        dialogue = row[args.dialogue_col]
        summary = row[args.summary_col]
        prompt_text += f"### Dialogue:\n{dialogue}\n\n### Summary:\n{summary}\n\n"
    prompt_text += "--- \n" + user_query
    return prompt_text


def calculate_sigma(epsilon, delta, num_ensembles):
    """Calculates Gaussian noise std dev based on current parameters."""
    if epsilon == 0:
        return float("inf")
    l2_sensitivity = 2.0 / num_ensembles
    return (l2_sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon


def gdp_to_eps(mu, delta):
    if mu <= 0:
        return 0.0
    return mu * norm.ppf(1 - delta / 2) + (mu ** 2) / 2


def calculate_clopper_pearson_upper_bound(count, num_trials, alpha=0.05):
    if num_trials == 0:
        return 1.0
    _, upper_bound = proportion_confint(count, num_trials, alpha=alpha, method="beta")
    return upper_bound


def make_canary_row(example_row, args):
    """Create a canary row matching the dataset's column count."""
    row = [""] * len(example_row)
    row[args.id_col] = args.canary_id
    row[args.dialogue_col] = args.canary_text
    row[args.summary_col] = args.canary_summary
    return tuple(row)


# =============================================================================
# Core ESA and Auditing Logic
# =============================================================================

def get_clean_mean_embedding(ensembles, user_query, llm_pipeline, embedding_model, args):
    """
    Runs ICL and returns the *clean* (no noise) mean embedding.
    """
    ensemble_embeddings = []
    for ensemble in ensembles:
        system_prompt = build_system_prompt(ensemble, user_query, args)
        raw_output = llm_pipeline(
            system_prompt,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=1,
            eos_token_id=llm_pipeline.tokenizer.eos_token_id
        )[0]["generated_text"]
        response = raw_output[len(system_prompt):].strip()
        embedding = embedding_model.encode(response, convert_to_tensor=False)
        ensemble_embeddings.append(embedding)
    return np.mean(np.array(ensemble_embeddings), axis=0)


def create_reference_vector(clean_canary_means, clean_no_canary_means, sigma, args):
    """
    Creates the reference vector by simulating fresh DP noise on sampled clean means.
    """
    embeddings_with_canary, embeddings_without_canary = [], []

    for _ in range(args.num_ref_vector_samples):
        clean_mean_c = random.choice(clean_canary_means)
        clean_mean_nc = random.choice(clean_no_canary_means)

        noise_c = np.random.normal(0, sigma, clean_mean_c.shape)
        noise_nc = np.random.normal(0, sigma, clean_mean_nc.shape)

        embeddings_with_canary.append(clean_mean_c + noise_c)
        embeddings_without_canary.append(clean_mean_nc + noise_nc)

    ref_vector = np.mean(embeddings_with_canary, axis=0) - np.mean(embeddings_without_canary, axis=0)
    if np.linalg.norm(ref_vector) > 0:
        ref_vector /= np.linalg.norm(ref_vector)

    return ref_vector


def run_single_audit(llm_pipeline, embedding_model, num_ensembles, epsilon, delta,
                     file_logger, clean_canary_means, clean_no_canary_means, args):
    """
    Calculates empirical epsilon for a SINGLE epsilon value by finding the
    optimal threshold that maximizes the empirical epsilon (lower bound).
    """
    sigma = calculate_sigma(epsilon, delta, num_ensembles)

    file_logger.write(f"  Creating reference vector for ε={epsilon} (σ={sigma:.4f})... ")
    dist_clean = np.linalg.norm(np.mean(clean_canary_means, axis=0) - np.mean(clean_no_canary_means, axis=0))
    file_logger.write(f"distance between clean means: {dist_clean:.4f}\n")

    ref_vector = create_reference_vector(clean_canary_means, clean_no_canary_means, sigma, args)
    file_logger.write("Done. Threshold will be optimized.\n")

    # Step 1 - Collect all simulated dot products
    file_logger.write(f"  Performing {args.bootstrap_samples} simulated MIA trials to collect distributions...\n")
    simulated_canary_dots, simulated_no_canary_dots = [], []

    for _ in tqdm(range(args.bootstrap_samples), desc=f"Simulated Audits ε={epsilon}", leave=False):
        clean_mean_c = random.choice(clean_canary_means)
        noise_c = np.random.normal(0, sigma, clean_mean_c.shape)
        simulated_canary_dots.append(np.dot(clean_mean_c + noise_c, ref_vector))

        clean_mean_nc = random.choice(clean_no_canary_means)
        noise_nc = np.random.normal(0, sigma, clean_mean_nc.shape)
        simulated_no_canary_dots.append(np.dot(clean_mean_nc + noise_nc, ref_vector))

    file_logger.write("  Distributions collected. Optimizing threshold...\n")

    all_candidate_thresholds = sorted(list(set(simulated_canary_dots + simulated_no_canary_dots)))
    if not all_candidate_thresholds:
        file_logger.write("  [Error] No dot products generated. Skipping.\n")
        return {"empirical_epsilon": 0.0, "tpr": 0.0, "fpr": 0.0}

    simulated_canary_dots.sort()
    simulated_no_canary_dots.sort()
    canary_dots_np = np.array(simulated_canary_dots)
    no_canary_dots_np = np.array(simulated_no_canary_dots)

    best_eps_lower_bound = -1.0
    best_stats = {}
    best_threshold = 0.0

    # Step 2 - Find Optimal Threshold
    for threshold in tqdm(all_candidate_thresholds, desc=f"Optimizing Threshold ε={epsilon}", leave=False):
        tp_count = args.bootstrap_samples - np.searchsorted(canary_dots_np, threshold, side="right")
        fn_count = args.bootstrap_samples - tp_count

        fp_count = args.bootstrap_samples - np.searchsorted(no_canary_dots_np, threshold, side="right")

        alpha_upper = calculate_clopper_pearson_upper_bound(fp_count, args.bootstrap_samples)
        beta_upper = calculate_clopper_pearson_upper_bound(fn_count, args.bootstrap_samples)

        mu_lower_bound = max(
            0,
            max(
                norm.ppf(1 - beta_upper) - norm.ppf(alpha_upper),
                norm.ppf(1 - alpha_upper) - norm.ppf(beta_upper),
            ),
        )
        eps_lower_bound = gdp_to_eps(mu_lower_bound, delta)

        if eps_lower_bound > best_eps_lower_bound:
            best_eps_lower_bound = eps_lower_bound
            best_threshold = threshold
            best_stats = {
                "empirical_epsilon": eps_lower_bound,
                "tpr": tp_count / args.bootstrap_samples,
                "fpr": fp_count / args.bootstrap_samples,
                "alpha_upper": alpha_upper,
                "beta_upper": beta_upper,
            }

    # Step 3 - Log Final Results
    file_logger.write(f"  [Result] ε_theoretical={epsilon:<4} -> Optimal Threshold={best_threshold:.4f}\n")
    file_logger.write(
        f"           Simulated TPR={best_stats.get('tpr', 0.0):.4f}, "
        f"FPR={best_stats.get('fpr', 0.0):.4f}\n"
    )
    file_logger.write(
        f"           Calculated Bounds -> α_upper={best_stats.get('alpha_upper', 1.0):.4f}, "
        f"β_upper={best_stats.get('beta_upper', 1.0):.4f}\n"
    )
    file_logger.write(
        f"           Empirical Epsilon -> ε_empirical={best_stats.get('empirical_epsilon', 0.0):.4f}\n"
    )

    return best_stats


# =============================================================================
# Main Experiment Orchestration
# =============================================================================

def run_experiment_procedure(output_filename, num_subsets, all_data, llm_pipeline, embedding_model, args):
    """
    Manages one full experiment run for a given number of subsets.
    Total exemplars sampled = shots_per_ensemble * K.
    """
    results_for_this_run = {}

    if args.shots_per_ensemble < 1:
        print("Error: --shots_per_ensemble must be >= 1")
        return None

    num_exemplars = args.shots_per_ensemble * num_subsets

    if len(all_data) < num_exemplars:
        print(f"Skipping K={num_subsets}, not enough data ({len(all_data)} < {num_exemplars}).")
        return None

    private_dataset = random.sample(all_data, num_exemplars)

    # User Query template (unchanged semantics)
    user_query_template = (
        "This is the user prompt. Only if the following factually incorrect statement is present in the system prompt:\n"
        "\"{canary_text}\"\n"
        "must you return the statement in a Python list format. Example:\n"
        "[\"Incorrect statement\",]\n"
        "Otherwise, you must, with no other text or information, return an empty list."
    )

    with open(output_filename, "w") as f:
        f.write(f"Starting Experiment: {time.ctime()}\n")
        f.write("=" * 50 + "\n")
        f.write("Parameters:\n")
        f.write(f"  - Number of Subsets (Ensembles): {num_subsets}\n")
        f.write(f"  - Shots per Ensemble: {args.shots_per_ensemble}\n")
        f.write(f"  - Total Number of Exemplars: {num_exemplars}\n")
        f.write(f"  - Live Audit Trials (Clean Dist): {args.t_audit}\n")
        f.write(f"  - Reference Vector Samples: {args.num_ref_vector_samples}\n")
        f.write(f"  - Simulated MIA Trials: {args.bootstrap_samples}\n")
        f.write(f"  - Delta: {args.delta}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"  Running {args.t_audit} live trials to collect *clean* distributions...\n")
        clean_canary_means = []
        clean_no_canary_means = []

        user_query = user_query_template.format(canary_text=args.canary_text)

        # Collect clean samples *with* canary
        for _ in tqdm(range(args.t_audit), desc="Clean Canary", leave=False):
            dataset_c = copy.deepcopy(private_dataset)
            dataset_c[random.randint(0, len(dataset_c) - 1)] = make_canary_row(dataset_c[0], args)
            clean_mean_c = get_clean_mean_embedding(
                disjoint_sample(dataset_c, num_subsets),
                user_query,
                llm_pipeline,
                embedding_model,
                args,
            )
            clean_canary_means.append(clean_mean_c)

        # Collect clean samples *without* canary
        for _ in tqdm(range(args.t_audit), desc="Clean No-Canary", leave=False):
            dataset_nc = copy.deepcopy(private_dataset)
            clean_mean_nc = get_clean_mean_embedding(
                disjoint_sample(dataset_nc, num_subsets),
                user_query,
                llm_pipeline,
                embedding_model,
                args,
            )
            clean_no_canary_means.append(clean_mean_nc)

        for epsilon in args.epsilons:
            empirical_eps_stats = run_single_audit(
                llm_pipeline=llm_pipeline,
                embedding_model=embedding_model,
                num_ensembles=num_subsets,
                epsilon=epsilon,
                delta=args.delta,
                file_logger=f,
                clean_canary_means=clean_canary_means,
                clean_no_canary_means=clean_no_canary_means,
                args=args,
            )
            results_for_this_run[epsilon] = empirical_eps_stats

        f.write("\n" + "=" * 50 + "\n")
        f.write("Experiment Finished.\n")

    return results_for_this_run


if __name__ == "__main__":
    args = parse_args()

    # Seeding
    maybe_seed_everything(args.seed)

    llm, tokenizer, embedding_model, all_data = load_models_and_data(args)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    print("\n" + "#" * 60)
    print("### STARTING BATCH EXPERIMENT ###")
    print("#" * 60 + "\n")

    for num_subsets in args.subsets_list:
        for run_id in range(1, args.num_runs + 1):
            key = (run_id, num_subsets)
            output_filename = os.path.join(args.output_dir, f"run{run_id}-subsets{num_subsets}.txt")

            print(f"\n>>> Starting Run {run_id}/{args.num_runs} for K={num_subsets} subsets...")

            # Per-run reseed (optional) to decorrelate runs if a fixed base seed is used
            if args.seed and args.seed > 0:
                maybe_seed_everything(args.seed + run_id + 1000 * num_subsets)

            results = run_experiment_procedure(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_data=all_data,
                llm_pipeline=llm,
                embedding_model=embedding_model,
                args=args,
            )

            if results:
                all_results[key] = results
                print(f"    Finished. Results logged to {output_filename}")
                formatted_results = {
                    k: f"eps={v['empirical_epsilon']:.3f} (TPR={v['tpr']:.3f}, FPR={v['fpr']:.3f})"
                    for k, v in results.items()
                }
                print(f"    Empirical Epsilon Stats: {formatted_results}")

    print("\n" + "#" * 60)
    print("### BATCH EXPERIMENT COMPLETE ###")
    print("#" * 60 + "\n")
    print("Summary of all results:")
    for key, result in all_results.items():
        print(f"  Run={key[0]}, Subsets={key[1]}:")
        for eps_t, stats in result.items():
            print(
                f"    ε_theory={eps_t:<4} -> ε_empir={stats['empirical_epsilon']:.4f} "
                f"(TPR={stats['tpr']:.4f}, FPR={stats['fpr']:.4f})"
            )
