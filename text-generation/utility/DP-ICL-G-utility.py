import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
import random
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import evaluate  # ROUGE
from sklearn.metrics.pairwise import cosine_similarity

# Ignore warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. Model & data loading
# =============================================================================

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}. Use bf16/fp16/fp32.")


def load_llm_and_tokenizer(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
):
    """Load LLM + tokenizer.

    By default uses 4-bit quantization to save VRAM (same as your original script).
    """
    print(f"Loading LLM + tokenizer from: {model_path} ...")

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # LLaMA-3 often needs a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        model.eval()
        print("LLM + tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        print("Please check model path and VRAM.")
        sys.exit(1)


def load_data(data_path: str, num_exemplars: int, num_test_samples: int, encoding: str = "latin-1"):
    """Load SAMSum-like dataset and split into exemplar_pool and test_data.

    NOTE: this script assumes CSV contains 'dialogue' and 'summary' columns.
    """
    print(f"Loading data from: {data_path} ...")
    try:
        df = pd.read_csv(data_path, encoding=encoding)

        if "dialogue" not in df.columns or "summary" not in df.columns:
            raise ValueError("CSV must contain 'dialogue' and 'summary' columns.")

        df = df.dropna(subset=["dialogue", "summary"])
        df = df[df["dialogue"].astype(str).str.strip() != ""]
        records = df.to_dict("records")
        random.shuffle(records)

        if len(records) < num_exemplars + num_test_samples:
            print(f"[WARN] Dataset too small: {len(records)} rows.")
            print("Using 80% exemplars, 20% test.")
            cut = int(len(records) * 0.8)
            exemplar_pool = records[:cut]
            test_data = records[cut:]
        else:
            exemplar_pool = records[:num_exemplars]
            test_data = records[num_exemplars: num_exemplars + num_test_samples]

        print(f"Loaded: {len(exemplar_pool)} exemplars, {len(test_data)} test samples.")
        return exemplar_pool, test_data

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def load_embedding_model(model_name: str, device: str | None = None):
    print(f"Loading embedding model: {model_name} ...")
    try:
        if device is None:
            model = SentenceTransformer(model_name)
        else:
            model = SentenceTransformer(model_name, device=device)
        print("Embedding model loaded.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")


# =============================================================================
# 2. Prompt formatting + generation
# =============================================================================

def format_prompt(query_dialogue, exemplars):
    prompt_lines = ["Summarize the following dialogue:\n"]
    for ex in exemplars:
        prompt_lines.append(f"---\nDialogue:\n{ex['dialogue']}\n")
        prompt_lines.append(f"Summary:\n{ex['summary']}\n")
    prompt_lines.append(f"---\nDialogue:\n{query_dialogue}\n")
    prompt_lines.append("Summary:\n")
    return "".join(prompt_lines)


def format_zeroshot_prompt(query_dialogue):
    return (
        "Summarize the following dialogue:\n"
        f"---\nDialogue:\n{query_dialogue}\n"
        "Summary:\n"
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_input_length: int,
    diverse: bool = False,
    temperature: float = 1.0,
):
    """Generate text with LLM.

    - diverse=True: sampling (for candidate generation)
    - diverse=False: greedy (for teacher model)
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if diverse:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0, input_length:]
    decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded_text.strip()


# =============================================================================
# 3. ESA core logic
# =============================================================================

def get_embedding(embed_model, text: str):
    """Generate L2-normalized embedding."""
    emb = embed_model.encode(text)
    norm = np.linalg.norm(emb)
    if norm > 0:
        return emb / norm
    return emb


def calculate_sigma(epsilon: float, delta: float, N: int):
    if epsilon == float("inf"):
        return 0.0
    sensitivity = 2.0 / N
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    return sigma


def run_esa_aggregation(
    query_dialogue: str,
    exemplar_pool,
    N: int,
    k_exemplars: int,
    llm,
    tokenizer,
    embed_model,
    epsilon: float,
    delta: float,
    max_new_tokens_summary: int,
    max_input_length: int,
    teacher_diverse: bool,
    teacher_temperature: float,
):
    """ESA steps 1-8: subset summaries -> mean embedding -> add Gaussian noise."""

    if len(exemplar_pool) < N * k_exemplars:
        print(f"[WARN] Exemplar pool ({len(exemplar_pool)}) < N*k ({N}*{k_exemplars}). Using sampling w/ replacement.")
        all_exemplars = [random.choice(exemplar_pool) for _ in range(N * k_exemplars)]
    else:
        all_exemplars = random.sample(exemplar_pool, N * k_exemplars)

    subsets = [all_exemplars[i * k_exemplars:(i + 1) * k_exemplars] for i in range(N)]

    embeddings = []
    for subset in subsets:
        prompt = format_prompt(query_dialogue, subset)
        summary_i = generate_text(
            llm,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens_summary,
            max_input_length=max_input_length,
            diverse=teacher_diverse,
            temperature=teacher_temperature,
        )
        emb_i = get_embedding(embed_model, summary_i)
        embeddings.append(emb_i)

    mean_emb = np.mean(embeddings, axis=0)

    sigma = calculate_sigma(epsilon, delta, N)
    if sigma > 0:
        noise = np.random.normal(loc=0.0, scale=sigma, size=mean_emb.shape)
        return mean_emb + noise
    return mean_emb


def get_candidates_and_select(
    query_dialogue: str,
    noisy_mean_emb: np.ndarray,
    num_candidates: int,
    llm,
    tokenizer,
    embed_model,
    max_new_tokens_candidate: int,
    max_input_length: int,
    candidate_temperature: float,
):
    """ESA step 9: generate zero-shot candidates and pick the most similar."""
    candidates = []
    candidate_embs = []

    for _ in range(num_candidates):
        prompt = format_zeroshot_prompt(query_dialogue)
        candidate_summary = generate_text(
            llm,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens_candidate,
            max_input_length=max_input_length,
            diverse=True,
            temperature=candidate_temperature,
        )
        cand_emb = get_embedding(embed_model, candidate_summary)
        candidates.append(candidate_summary)
        candidate_embs.append(cand_emb)

    if not candidates:
        return ""

    noisy_emb_reshaped = noisy_mean_emb.reshape(1, -1)
    embs_array = np.array(candidate_embs)
    similarities = cosine_similarity(noisy_emb_reshaped, embs_array)[0]
    best_index = int(np.argmax(similarities))
    return candidates[best_index]


# =============================================================================
# 4. Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DP-ICL ESA Utility evaluation (ROUGE) for SAMSum-like datasets.")

    # Paths (parameterized; defaults match your original constants)
    parser.add_argument("--llm_path", type=str, default="/local/scratch/yxia327/Meta-Llama-3-8B-Instruct-HF")
    parser.add_argument("--data_path", type=str, default="/local/scratch/yxia327/data/Samsum/train.csv")
    parser.add_argument("--data_encoding", type=str, default="latin-1")
    parser.add_argument("--embedding_model_name", type=str, default="/local/scratch/yxia327/bge-large-en-v1.5")
    parser.add_argument("--embedding_device", type=str, default=None, help="e.g., 'cuda' or 'cpu'")

    # LLM loading (bnb 4bit)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bf16", help="bf16|fp16|fp32")

    # Experiment grid (keep original comma-separated UX)
    parser.add_argument("--num_subsets", type=str, default="2,4,6,8,10,12,14",
                        help="Comma-separated N values (subsets), e.g. '2,4,6'")
    parser.add_argument("--epsilons", type=str, default="1,2,4,8",
                        help="Comma-separated epsilons; use 'inf' for no-noise, e.g. '1,2,inf'")

    # Dataset split / evaluation
    parser.add_argument("--num_exemplars_total", type=int, default=1000)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--k_exemplars", type=int, default=1, help="k-shot per subset")
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-5)

    # Generation knobs
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens_summary", type=int, default=150)
    parser.add_argument("--max_new_tokens_candidate", type=int, default=150)

    # IMPORTANT: original script generated subset summaries with sampling (diverse=True).
    # Keep as default to preserve behavior.
    parser.add_argument("--teacher_diverse", action="store_true", default=True,
                        help="Use sampling for subset summaries (default True: matches original behavior).")
    parser.add_argument("--teacher_greedy", action="store_false", dest="teacher_diverse",
                        help="Use greedy decoding for subset summaries.")
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--candidate_temperature", type=float, default=1.0)

    # Reproducibility / output
    parser.add_argument("--seed", type=int, default=0, help="0 = do not force seeding")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional: save raw results to CSV path")

    return parser.parse_args()


def set_seed(seed: int):
    if seed and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    list_num_subsets = [int(n.strip()) for n in args.num_subsets.split(",") if n.strip()]
    list_epsilons = []
    for e in args.epsilons.split(","):
        e = e.strip()
        if not e:
            continue
        list_epsilons.append(float("inf") if e.lower() == "inf" else float(e))

    print("--- Config ---")
    print(f"LLM path: {args.llm_path}")
    print(f"Data path: {args.data_path}")
    print(f"Embedding model: {args.embedding_model_name} (device={args.embedding_device})")
    print(f"N list: {list_num_subsets}")
    print(f"Epsilon list: {list_epsilons}")
    print(f"Delta: {args.delta}")
    print(f"k-shot per subset: {args.k_exemplars}")
    print(f"Exemplar pool size: {args.num_exemplars_total}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"Candidates: {args.num_candidates}")
    print(f"Teacher decoding: {'sampling' if args.teacher_diverse else 'greedy'} (temp={args.teacher_temperature})")
    print(f"Candidate temp: {args.candidate_temperature}")
    print("----------------")

    llm, tokenizer = load_llm_and_tokenizer(
        model_path=args.llm_path,
        device_map=args.device_map,
        torch_dtype=_dtype_from_str(args.torch_dtype),
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_str(args.bnb_4bit_compute_dtype),
    )
    embed_model = load_embedding_model(args.embedding_model_name, device=args.embedding_device)
    exemplar_pool, test_data = load_data(
        args.data_path,
        args.num_exemplars_total,
        args.num_test_samples,
        encoding=args.data_encoding,
    )
    rouge = evaluate.load("rouge")

    results = []

    for N in list_num_subsets:
        for epsilon in list_epsilons:
            print(f"\n--- Running: N={N}, epsilon={epsilon} ---")

            predictions = []
            references = []

            pbar = tqdm(test_data, desc=f"N={N}, Eps={epsilon}")
            for sample in pbar:
                query_dialogue = sample["dialogue"]
                reference_summary = sample["summary"]

                noisy_mean_emb = run_esa_aggregation(
                    query_dialogue=query_dialogue,
                    exemplar_pool=exemplar_pool,
                    N=N,
                    k_exemplars=args.k_exemplars,
                    llm=llm,
                    tokenizer=tokenizer,
                    embed_model=embed_model,
                    epsilon=epsilon,
                    delta=args.delta,
                    max_new_tokens_summary=args.max_new_tokens_summary,
                    max_input_length=args.max_input_length,
                    teacher_diverse=args.teacher_diverse,
                    teacher_temperature=args.teacher_temperature,
                )

                final_summary = get_candidates_and_select(
                    query_dialogue=query_dialogue,
                    noisy_mean_emb=noisy_mean_emb,
                    num_candidates=args.num_candidates,
                    llm=llm,
                    tokenizer=tokenizer,
                    embed_model=embed_model,
                    max_new_tokens_candidate=args.max_new_tokens_candidate,
                    max_input_length=args.max_input_length,
                    candidate_temperature=args.candidate_temperature,
                )

                predictions.append(final_summary)
                references.append(reference_summary)

            try:
                rouge_scores = rouge.compute(predictions=predictions, references=references)
                rouge_l_f1 = float(rouge_scores["rougeL"])
            except Exception as e:
                print(f"[WARN] ROUGE compute error: {e}")
                rouge_l_f1 = 0.0

            print(f"Result: N={N}, epsilon={epsilon} -> ROUGE-L={rouge_l_f1:.4f}")
            results.append({"N (Subsets)": N, "Epsilon": epsilon, "ROUGE-L": rouge_l_f1})

    print("\n\n--- Summary ---")
    results_df = pd.DataFrame(results)

    try:
        pivot_df = results_df.pivot(index="N (Subsets)", columns="Epsilon", values="ROUGE-L")
        pivot_df.columns.name = "Epsilon"
        print(pivot_df.to_string(float_format="%.4f"))
    except Exception as e:
        print(f"[WARN] Pivot failed ({e}). Printing raw results:")
        print(results_df.to_string(index=False))

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        results_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved results CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()
