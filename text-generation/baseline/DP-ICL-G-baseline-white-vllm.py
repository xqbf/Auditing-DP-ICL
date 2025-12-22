import argparse
import csv
import os
import random
import copy
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "DP-ICL ESA baseline (vLLM black-box) with simulated MIA threshold optimization.\n"
            "NOTE: Default assumes a SAMSum-like CSV with columns (id, dialogue, summary).\n"
            "You can run other datasets ONLY IF you set the correct column indices."
        )
    )

    # vLLM client
    p.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--model_id", type=str, default="/local/scratch/yxia327/gpt-oss-20b",
                   help="Model name/path served by vLLM. Must match server-side `--model`.")
    p.add_argument("--max_tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)

    # Embedding model
    p.add_argument("--embedding_model_name", type=str, default="/local/scratch/yxia327/bge-large-en-v1.5")
    p.add_argument("--embedding_device", type=str, default="cuda")

    # Data
    p.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/Samsum/train.csv")
    p.add_argument("--data_encoding", type=str, default="latin-1")
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--has_header", action="store_true", default=True)
    p.add_argument("--no_header", action="store_false", dest="has_header")

    # Column indices (default: SAMSum)
    p.add_argument("--id_col", type=int, default=0)
    p.add_argument("--dialogue_col", type=int, default=1)
    p.add_argument("--summary_col", type=int, default=2)

    # Experiment parameters
    p.add_argument("--subsets_list", type=int, nargs="+", default=[4])
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--num_runs", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="experiment_results_baseline")
    p.add_argument("--output_prefix", type=str, default="gpt-run")

    # Shot control
    p.add_argument("--shots_per_ensemble", type=int, default=1,
                   help="Shots per ensemble. Total exemplars = shots_per_ensemble * K. Default=1 (original behavior).")

    # Audit parameters
    p.add_argument("--t_audit", type=int, default=400)
    p.add_argument("--num_ref_vector_samples", type=int, default=200)
    p.add_argument("--bootstrap_samples", type=int, default=400000)
    p.add_argument("--delta", type=float, default=1e-5)

    # Canary / prompt
    p.add_argument("--canary_text", type=str, default="The sun rises in the west.")
    p.add_argument("--canary_id", type=str, default="canary_id")
    p.add_argument("--canary_summary", type=str, default="canary_summary")

    # Reproducibility
    p.add_argument("--seed", type=int, default=0, help="0 = do not force seeding.")

    return p.parse_args()


def maybe_seed(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        np.random.seed(seed)


def load_embedding_and_data(args):
    print("Loading embedding model and data...")

    embedding_model = SentenceTransformer(args.embedding_model_name, device=args.embedding_device)

    with open(args.data_path, "r", encoding=args.data_encoding) as f:
        rows = [tuple(r) for r in csv.reader(f, delimiter=args.delimiter)]
    if args.has_header and len(rows) > 0:
        rows = rows[1:]
    all_data = rows

    if len(all_data) == 0:
        raise RuntimeError("Dataset is empty after loading.")

    max_idx = max(args.id_col, args.dialogue_col, args.summary_col)
    if len(all_data[0]) <= max_idx:
        raise RuntimeError(f"Row has {len(all_data[0])} cols but requested index {max_idx}.")

    print("Embedding model and data loaded successfully.")
    return embedding_model, all_data


def disjoint_sample(dataset, m):
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
    row = [""] * len(example_row)
    row[args.id_col] = args.canary_id
    row[args.dialogue_col] = args.canary_text
    row[args.summary_col] = args.canary_summary
    return tuple(row)


USER_QUERY_TEMPLATE = (
    "This is the user prompt. Only if the following factually incorrect statement is present in the system prompt:\n"
    "\"{canary_text}\"\n"
    "must you return the statement in a Python list format. Example:\n"
    "[\"Incorrect statement\",]\n"
    "Otherwise, you must, with no other text or information, return an empty list."
)


def get_clean_mean_embedding(ensembles, user_query, embedding_model, client: OpenAI, args):
    ensemble_embeddings = []
    for ensemble in ensembles:
        system_prompt = build_system_prompt(ensemble, user_query, args)
        try:
            completion = client.chat.completions.create(
                model=args.model_id,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=None,
            )
            response = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] vLLM API error: {e}")
            response = ""
        embedding = embedding_model.encode(response, convert_to_tensor=False)
        ensemble_embeddings.append(embedding)
    return np.mean(np.array(ensemble_embeddings), axis=0)


def create_reference_vector(clean_canary_means, clean_no_canary_means, sigma, args):
    embeddings_with_canary, embeddings_without_canary = [], []
    for _ in range(args.num_ref_vector_samples):
        clean_mean_c = random.choice(clean_canary_means)
        clean_mean_nc = random.choice(clean_no_canary_means)
        embeddings_with_canary.append(clean_mean_c + np.random.normal(0, sigma, clean_mean_c.shape))
        embeddings_without_canary.append(clean_mean_nc + np.random.normal(0, sigma, clean_mean_nc.shape))

    ref_vector = np.mean(embeddings_with_canary, axis=0) - np.mean(embeddings_without_canary, axis=0)
    if np.linalg.norm(ref_vector) > 0:
        ref_vector /= np.linalg.norm(ref_vector)
    return ref_vector


def run_single_audit(epsilon, file_logger, clean_canary_means, clean_no_canary_means, num_ensembles, args):
    sigma = calculate_sigma(epsilon, args.delta, num_ensembles)

    file_logger.write(f"  Creating reference vector for ε={epsilon} (σ={sigma:.4f})... ")
    ref_vector = create_reference_vector(clean_canary_means, clean_no_canary_means, sigma, args)
    file_logger.write("Done. Threshold will be optimized.\n")

    file_logger.write(f"  Performing {args.bootstrap_samples} simulated MIA trials to collect distributions...\n")
    simulated_canary_dots, simulated_no_canary_dots = [], []

    for _ in tqdm(range(args.bootstrap_samples), desc=f"Simulated Audits ε={epsilon}", leave=False):
        clean_mean_c = random.choice(clean_canary_means)
        simulated_canary_dots.append(np.dot(clean_mean_c + np.random.normal(0, sigma, clean_mean_c.shape), ref_vector))

        clean_mean_nc = random.choice(clean_no_canary_means)
        simulated_no_canary_dots.append(np.dot(clean_mean_nc + np.random.normal(0, sigma, clean_mean_nc.shape), ref_vector))

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
        eps_lower_bound = gdp_to_eps(mu_lower_bound, args.delta)

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


def run_experiment_procedure(output_filename, num_subsets, all_data, embedding_model, client: OpenAI, args):
    results_for_this_run = {}

    if args.shots_per_ensemble < 1:
        print("Error: --shots_per_ensemble must be >= 1")
        return None

    num_exemplars = args.shots_per_ensemble * num_subsets
    if len(all_data) < num_exemplars:
        print(f"Skipping K={num_subsets}, not enough data ({len(all_data)} < {num_exemplars}).")
        return None

    private_dataset = random.sample(all_data, num_exemplars)

    with open(output_filename, "w") as f:
        f.write(f"Starting Experiment: {time.ctime()}\n")
        f.write("=" * 50 + "\n")
        f.write("Parameters:\n")
        f.write(f"  - vLLM base_url: {args.base_url}\n")
        f.write(f"  - vLLM model_id: {args.model_id}\n")
        f.write(f"  - Number of Subsets (Ensembles): {num_subsets}\n")
        f.write(f"  - Shots per Ensemble: {args.shots_per_ensemble}\n")
        f.write(f"  - Total Number of Exemplars: {num_exemplars}\n")
        f.write(f"  - Live Audit Trials (Clean Dist): {args.t_audit}\n")
        f.write(f"  - Reference Vector Samples: {args.num_ref_vector_samples}\n")
        f.write(f"  - Simulated MIA Trials: {args.bootstrap_samples}\n")
        f.write(f"  - Delta: {args.delta}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"  Running {args.t_audit} live trials to collect *clean* distributions...\n")
        clean_canary_means, clean_no_canary_means = [], []

        user_query = USER_QUERY_TEMPLATE.format(canary_text=args.canary_text)

        for _ in tqdm(range(args.t_audit), desc="Clean Canary", leave=False):
            dataset_c = copy.deepcopy(private_dataset)
            dataset_c[random.randint(0, len(dataset_c) - 1)] = make_canary_row(dataset_c[0], args)
            clean_canary_means.append(
                get_clean_mean_embedding(disjoint_sample(dataset_c, num_subsets), user_query, embedding_model, client, args)
            )

        for _ in tqdm(range(args.t_audit), desc="Clean No-Canary", leave=False):
            dataset_nc = copy.deepcopy(private_dataset)
            clean_no_canary_means.append(
                get_clean_mean_embedding(disjoint_sample(dataset_nc, num_subsets), user_query, embedding_model, client, args)
            )

        for epsilon in args.epsilons:
            stats = run_single_audit(
                epsilon=epsilon,
                file_logger=f,
                clean_canary_means=clean_canary_means,
                clean_no_canary_means=clean_no_canary_means,
                num_ensembles=num_subsets,
                args=args,
            )
            results_for_this_run[epsilon] = stats

        f.write("\n" + "=" * 50 + "\n")
        f.write("Experiment Finished.\n")

    return results_for_this_run


def main():
    args = parse_args()
    maybe_seed(args.seed)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)
    embedding_model, all_data = load_embedding_and_data(args)

    print("\n" + "#" * 60)
    print("### STARTING BATCH EXPERIMENT ###")
    print(f"### Target Model (vLLM): {args.model_id} ###")
    print("#" * 60 + "\n")

    all_results = {}
    for num_subsets in args.subsets_list:
        for run_id in range(1, args.num_runs + 1):
            if args.seed and args.seed > 0:
                maybe_seed(args.seed + run_id + 1000 * num_subsets)

            output_filename = os.path.join(args.output_dir, f"{args.output_prefix}{run_id}-subsets{num_subsets}.txt")
            print(f"\n>>> Starting Run {run_id}/{args.num_runs} for K={num_subsets} subsets...")

            results = run_experiment_procedure(
                output_filename=output_filename,
                num_subsets=num_subsets,
                all_data=all_data,
                embedding_model=embedding_model,
                client=client,
                args=args,
            )
            if results:
                all_results[(run_id, num_subsets)] = results
                print(f"    Finished. Results logged to {output_filename}")

    print("\n" + "#" * 60)
    print("### BATCH EXPERIMENT COMPLETE ###")
    print("#" * 60 + "\n")
    for (run_id, k), result in all_results.items():
        print(f"  Run={run_id}, Subsets={k}:")
        for eps_t, stats in result.items():
            print(
                f"    ε_theory={eps_t:<4} -> ε_empir={stats['empirical_epsilon']:.4f} "
                f"(TPR={stats['tpr']:.4f}, FPR={stats['fpr']:.4f})"
            )


if __name__ == "__main__":
    main()
