"""
MSGCG: Multi-Sweep GCG for Maximally Distant Sentence Pairs.

Finds signal sentences (y1, y2) for ESA whose BGE embeddings are as far apart as
possible. It is MSGS with the greedy word search replaced by GCG (Zou et al. 2023):

  1. Start from a natural seed sentence A and a control sequence B = "! ! ... !".
  2. Sweep (single-sweep GCG): fix one endpoint as the anchor and optimise the other.
     Each step takes the gradient of cos(anchor, moving) with respect to the moving
     sentence's one-hot tokens, samples one-position replacements from the top-k
     tokens, keeps candidates that survive decode/re-tokenize, and moves to the best.
  3. Multi-sweep: alternate the endpoints (optimise B against A, then A against B, ...)
     until a sweep no longer increases the distance.
  4. Multi-restart: repeat for several seed sentences and keep the best pairs.

Tokens are restricted to alphabetic wordpieces so the LLM can copy the output.

Usage:
    python msgcg.py --embedding_model /path/to/bge-large-en-v1.5
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


SEED_SENTENCES = [
    "The children played happily in the park after school.",
    "She made a cup of coffee and sat down to read the newspaper.",
    "The train arrived at the station five minutes ahead of schedule.",
    "He walked his dog along the quiet riverside path every morning.",
    "The restaurant on the corner serves excellent Italian food.",
    "They spent the weekend painting the living room a soft blue.",
    "The library was warm and quiet on that cold winter afternoon.",
    "My neighbor grows beautiful roses in her front garden every summer.",
]


# ==============================================================================
# GCG building blocks
# ==============================================================================
def cls_embedding(model, input_ids):
    hidden = model(input_ids=input_ids).last_hidden_state
    return F.normalize(hidden[:, 0], dim=-1)


def token_gradient(model, weights, ids, cls_id, sep_id, anchor):
    """Gradient of cos(anchor, moving) with respect to the moving sentence's one-hot tokens."""
    one_hot = torch.zeros(ids.shape[0], weights.shape[0], device=ids.device, dtype=weights.dtype)
    one_hot.scatter_(1, ids.unsqueeze(1), 1.0)
    one_hot.requires_grad_()
    special = weights[torch.tensor([cls_id, sep_id], device=ids.device)].detach()
    embeds = torch.cat([special[:1], one_hot @ weights, special[1:]]).unsqueeze(0)
    hidden = model(inputs_embeds=embeds).last_hidden_state
    cosine = (F.normalize(hidden[:, 0], dim=-1) * anchor).sum()
    return torch.autograd.grad(cosine, one_hot)[0]


def sample_candidates(ids, grad, disallowed, width, topk):
    grad = grad.clone()
    grad[:, disallowed] = float("inf")
    top = (-grad).topk(topk, dim=1).indices
    positions = torch.linspace(0, len(ids) - 1, width, device=ids.device).round().long()
    picks = top[positions, torch.randint(0, topk, (width,), device=ids.device)]
    candidates = ids.repeat(width, 1)
    candidates[torch.arange(width, device=ids.device), positions] = picks
    return candidates


def keep_decodable(tokenizer, candidates, current):
    """Candidates whose decoded text re-tokenizes to the same length and differs from current."""
    kept = []
    for row in candidates.tolist():
        text = tokenizer.decode(row, skip_special_tokens=True)
        again = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(again) == len(row) and again != current:
            kept.append(again)
    return kept


def gcg_sweep(model, tokenizer, weights, anchor_text, moving_ids, disallowed, args, device):
    """One single-sweep GCG pass: move `moving_ids` away from the fixed anchor sentence.
    Returns (best moving text, best cosine)."""
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    with torch.no_grad():
        anchor_ids = tokenizer(anchor_text, return_tensors="pt")["input_ids"].to(device)
        anchor = cls_embedding(model, anchor_ids)
        start = torch.cat([torch.tensor([[cls_id]], device=device), moving_ids.unsqueeze(0),
                           torch.tensor([[sep_id]], device=device)], dim=1)
        best_cos = float((cls_embedding(model, start) * anchor).sum())
    ids, best_ids = moving_ids.clone(), moving_ids.clone()

    for step in range(args.steps):
        grad = token_gradient(model, weights, ids, cls_id, sep_id, anchor)
        candidates = sample_candidates(ids, grad, disallowed, args.search_width, args.topk)
        kept = keep_decodable(tokenizer, candidates, ids.tolist())
        if not kept:
            continue
        batch = torch.tensor(kept, device=device)
        wrapped = torch.cat([torch.full((len(kept), 1), cls_id, device=device), batch,
                             torch.full((len(kept), 1), sep_id, device=device)], dim=1)
        with torch.no_grad():
            cosines = (cls_embedding(model, wrapped) * anchor).sum(dim=1)
        index = int(cosines.argmin())
        ids = batch[index]
        if float(cosines[index]) < best_cos:
            best_cos, best_ids = float(cosines[index]), ids.clone()
        if (step + 1) % 10 == 0:
            print(f"    step {step + 1}: dist={np.sqrt(max(2 - 2 * best_cos, 0.0)):.4f}", flush=True)
    return tokenizer.decode(best_ids.tolist(), skip_special_tokens=True), best_cos


# ==============================================================================
# Multi-sweep: alternate the fixed endpoint
# ==============================================================================
def find_distant_pair(model, tokenizer, weights, seed_sentence, disallowed, args, device):
    def to_ids(text):
        return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], device=device)

    sent_a = seed_sentence
    sent_b = " ".join(["!"] * args.control_len)
    best_dist = 0.0

    for sweep in range(args.max_sweeps):
        # Even sweeps move B against A, odd sweeps move A against B.
        anchor, moving = (sent_a, sent_b) if sweep % 2 == 0 else (sent_b, sent_a)
        new_text, cos = gcg_sweep(model, tokenizer, weights, anchor, to_ids(moving),
                                  disallowed, args, device)
        dist = float(np.sqrt(max(2 - 2 * cos, 0.0)))
        print(f"  Sweep {sweep} ({'B' if sweep % 2 == 0 else 'A'} moves): pair distance = {dist:.6f}",
              flush=True)
        if dist <= best_dist + 1e-7:
            print(f"  Sweep {sweep}: converged at {best_dist:.6f}", flush=True)
            break
        best_dist = dist
        if sweep % 2 == 0:
            sent_b = new_text
        else:
            sent_a = new_text
    return sent_a, sent_b


def main():
    p = argparse.ArgumentParser(description="MSGCG: multi-sweep GCG distant sentence search")
    p.add_argument("--embedding_model", required=True)
    p.add_argument("--control_len", type=int, default=16)
    p.add_argument("--steps", type=int, default=60, help="GCG steps per sweep")
    p.add_argument("--max_sweeps", type=int, default=10, help="Max alternating sweeps per seed")
    p.add_argument("--search_width", type=int, default=128)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--num_seeds", type=int, default=len(SEED_SENTENCES),
                   help="Number of seed sentences (restarts) to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default="msgcg_distant_pairs.json")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    weights = model.get_input_embeddings().weight
    vocab = tokenizer.get_vocab()
    disallowed = torch.tensor([i for tok, i in vocab.items()
                               if not (tok.removeprefix("##").isascii() and tok.removeprefix("##").isalpha())
                               or i in tokenizer.all_special_ids], device=device)

    reference = SentenceTransformer(args.embedding_model, device=device)
    results = []
    for rank, seed_sentence in enumerate(SEED_SENTENCES[:args.num_seeds]):
        start = time.time()
        print(f"Seed {rank}: {seed_sentence}", flush=True)
        sent_a, sent_b = find_distant_pair(model, tokenizer, weights, seed_sentence,
                                           disallowed, args, device)
        # Distance through the same encoder ESA uses, on the decoded strings.
        a, b = reference.encode([sent_a, sent_b], convert_to_numpy=True, normalize_embeddings=True)
        distance = float(np.linalg.norm(a - b))
        print(f"  final distance={distance:.6f} ({time.time() - start:.0f}s)\n"
              f"  S1: {sent_a}\n  S2: {sent_b}", flush=True)
        results.append({"seed_rank": rank, "euclidean_distance": distance,
                        "sentence_1": sent_a, "sentence_2": sent_b})

    results.sort(key=lambda r: -r["euclidean_distance"])
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
