"""
Shared utilities for text-generation auditing (ESA and KSA).
Includes: LLM setup, DP estimators, error bounds, data loading.
"""

import numpy as np
import math
import csv
import sys
import torch
from scipy.stats import beta, norm
from scipy.optimize import brentq
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ==============================================================================
# Global model state
# ==============================================================================
tokenizer = None
model = None
attack_mode = "semantic_similarity"
max_new_tokens_global = 10


def load_hf_model(model_path: str):
    global tokenizer, model
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


# ==============================================================================
# Analytic Gaussian Mechanism (Balle & Wang 2018)
# ==============================================================================
def _analytic_gauss_delta(sigma, sensitivity, epsilon):
    if sigma <= 0:
        return float('inf')
    t1 = norm.cdf(sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity)
    t2 = math.exp(epsilon) * norm.cdf(-sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity)
    return t1 - t2


def analytic_gauss_sigma(sensitivity, epsilon, delta):
    if epsilon <= 0:
        return float('inf')
    if delta >= 1:
        return 0.0
    sigma_classic = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    sigma_lo, sigma_hi = 1e-12, sigma_classic * 2.0
    if _analytic_gauss_delta(sigma_hi, sensitivity, epsilon) > delta:
        return sigma_classic
    for _ in range(200):
        sigma_mid = (sigma_lo + sigma_hi) / 2.0
        if _analytic_gauss_delta(sigma_mid, sensitivity, epsilon) > delta:
            sigma_lo = sigma_mid
        else:
            sigma_hi = sigma_mid
    return sigma_hi


# ==============================================================================
# Prompt building (membership query)
# ==============================================================================
def build_membership_prompt(context_text: str, query_text: str) -> str:
    if attack_mode == "direct":
        return f"""You are a highly accurate text verification system. Your sole function is to verify if the text in the <query> tag is present in the <context> tag.
Your response must be a single word: either 'Yes' or 'No'.

<context>
- {context_text}
</context>

<query>
{query_text}
</query>

<response>
"""
    elif attack_mode == "semantic_similarity":
        return f"""Determine whether the query has a near-exact semantic match with the reference text below.
Output only 'Yes' or 'No'.

Reference text:
- {context_text}

Query:
{query_text}

Answer:"""
    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")


def generate_response(prompt: str) -> str:
    """Generate text response from LLM."""
    messages = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(raw_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens_global,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def parse_yes_no(response: str) -> str:
    r = response.lower().strip()
    if r.startswith("yes"):
        return "Yes"
    if r.startswith("no"):
        return "No"
    if "yes" in r:
        return "Yes"
    if "no" in r:
        return "No"
    return "No"


def build_esa_generator_prompt(canary: str, y1_target: str, y2_control: str,
                               exemplar_context: str, rand_seed: int = None) -> str:
    """Build ESA constrained generation prompt (matches paper Appendix D.2).

    When exemplar_context is empty (0-shot candidate generation), a random seed
    is injected to break determinism at temperature 0.
    """
    if rand_seed is None:
        rand_seed = np.random.randint(0, 2**31)
    return f"""CONTEXTUAL EXEMPLAR:
{exemplar_context}

You have been given a CONTEXTUAL EXEMPLAR. If the CONTEXTUAL EXEMPLAR is empty, output exactly one of "{y1_target}" or "{y2_control}", chosen using random seed "{rand_seed}".
Otherwise, carefully check whether the dialog "{canary}" is present in the provided CONTEXTUAL EXEMPLAR.
If "{canary}" is found, output "{y1_target}".
Otherwise, output "{y2_control}".

RESPONSE:
"""


# ==============================================================================
# Statistical tools
# ==============================================================================
def clopper_pearson(successes, trials, confidence_level=0.95):
    if trials == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence_level
    lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (lower if not np.isnan(lower) else 0.0, upper if not np.isnan(upper) else 1.0)


def gdp_to_eps(mu, delta=1e-5):
    if mu <= 0:
        return 0.0
    def f(eps):
        t1 = norm.cdf(-eps / mu + mu / 2)
        log_t2 = eps + norm.logcdf(-eps / mu - mu / 2)
        t2 = np.exp(log_t2) if log_t2 < 500 else np.inf
        return t1 - t2 - delta
    try:
        return brentq(f, 0, 500)
    except ValueError:
        return 0.0


def pure_dp_eps_lower_bound(alpha_upper, beta_upper, delta=0.0):
    eps = 0.0
    if alpha_upper > 0 and (1.0 - beta_upper - delta) > 0:
        eps = max(eps, math.log((1.0 - beta_upper - delta) / alpha_upper))
    if beta_upper > 0 and (1.0 - alpha_upper - delta) > 0:
        eps = max(eps, math.log((1.0 - alpha_upper - delta) / beta_upper))
    return eps


def empirical_bernstein_correction(sigma2, n_llm, n_sample, gamma=0.05):
    """
    Two-layer error correction Delta_b (Algorithm 2).
    Delta_b = sqrt(2*sigma2*log(8/gamma)/n_llm) + 7*log(8/gamma)/(3*(n_llm-1)) + sqrt(log(8/gamma)/(2*n_sample))
    """
    log_term = math.log(8.0 / gamma)
    bernstein = math.sqrt(2.0 * sigma2 * log_term / n_llm) + 7.0 * log_term / (3.0 * (n_llm - 1))
    mc = math.sqrt(log_term / (2.0 * n_sample))
    return bernstein + mc


# ==============================================================================
# Data loading
# ==============================================================================
DATASET_CONFIGS = {
    "samsum": {"text_cols": [1, 2], "has_header": True, "encoding": "latin-1"},
    "xsum": {"text_cols": [1, 2], "has_header": True, "encoding": "utf-8"},
}


def load_texts(data_path: str, dataset_name: str) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    config = DATASET_CONFIGS[dataset_name]
    texts = []
    with open(data_path, "r", encoding=config.get("encoding", "utf-8")) as f:
        reader = csv.reader(f, delimiter=",")
        if config.get("has_header", False):
            next(reader, None)
        for row in reader:
            if not row:
                continue
            parts = []
            for col in config["text_cols"]:
                if col < len(row):
                    parts.append(str(row[col]).strip())
            text = " ".join(parts)
            if text:
                texts.append(text)
    return texts


def load_data_csv(data_path, encoding="latin-1", has_header=True, delimiter=","):
    """Load raw CSV rows for ESA (needs dialogue + summary columns)."""
    csv.field_size_limit(10 * 1024 * 1024)
    with open(data_path, "r", encoding=encoding) as f:
        rows = [tuple(r) for r in csv.reader(f, delimiter=delimiter)]
    if has_header and len(rows) > 0:
        rows = rows[1:]
    return rows
