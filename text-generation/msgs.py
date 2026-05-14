"""
MSGS: Multi-Sweep Greedy Search for Maximally Distant Sentence Pairs.

Greedy iterative word-replacement search for maximally distant sentence pairs
in embedding space. Used to find optimal signal sentences (y1, y2) for ESA.

Algorithm:
  1. Start with a seed sentence.
  2. Inner loop: greedily replace one word at a time to maximize Euclidean
     distance from the anchor embedding.
  3. Outer loop (multi-sweep): alternate optimising endpoints A and B.
  4. Multi-restart: try multiple seed sentences, keep the best pairs.

Usage:
    python msgs.py --embedding_model /path/to/bge-large-en-v1.5
"""

import argparse
import heapq
import json
import random
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ==============================================================================
# Helpers
# ==============================================================================
def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def embed_sentence(model, text):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(model, texts, batch_size=256):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)


# ==============================================================================
# Vocabulary
# ==============================================================================
def load_vocabulary(vocab_source, max_vocab):
    if vocab_source == "nltk":
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words
        vocab = list(set(w.lower() for w in words.words()))
    else:
        with open(vocab_source, "r") as f:
            vocab = [line.strip() for line in f if line.strip()]

    vocab = list(set(vocab))
    if max_vocab and len(vocab) > max_vocab:
        random.shuffle(vocab)
        vocab = vocab[:max_vocab]
    return vocab


# ==============================================================================
# Core algorithm
# ==============================================================================
def greedy_word_replace(model, anchor_emb, sentence, vocab,
                        max_inner_steps, top_k_positions=3, top_k_words=50,
                        verbose=False):
    words = sentence.split()
    current_emb = embed_sentence(model, sentence)
    current_dist = euclidean_distance(anchor_emb, current_emb)

    for step in range(max_inner_steps):
        best_step_sentence = None
        best_step_emb = None
        best_step_dist = current_dist

        if len(words) <= top_k_positions:
            candidate_positions = list(range(len(words)))
        else:
            removal_sentences = []
            for i in range(len(words)):
                removed = words[:i] + words[i + 1:]
                removal_sentences.append(" ".join(removed))
            removal_embs = embed_batch(model, removal_sentences)
            removal_dists = np.linalg.norm(removal_embs - anchor_emb[None, :], axis=1)
            candidate_positions = np.argsort(-removal_dists)[:top_k_positions].tolist()

        for pos in candidate_positions:
            candidates = random.sample(vocab, min(top_k_words, len(vocab)))
            trial_sentences = []
            for w in candidates:
                trial = words.copy()
                trial[pos] = w
                trial_sentences.append(" ".join(trial))

            trial_embs = embed_batch(model, trial_sentences)
            trial_dists = np.linalg.norm(trial_embs - anchor_emb[None, :], axis=1)
            best_idx = int(np.argmax(trial_dists))

            if trial_dists[best_idx] > best_step_dist:
                best_step_dist = trial_dists[best_idx]
                best_step_sentence = trial_sentences[best_idx]
                best_step_emb = trial_embs[best_idx]

        if best_step_sentence is None or best_step_dist <= current_dist + 1e-7:
            if verbose:
                print(f"  Inner step {step}: no improvement, stopping.")
            break

        words = best_step_sentence.split()
        current_emb = best_step_emb
        current_dist = best_step_dist
        if verbose:
            print(f"  Inner step {step}: dist={current_dist:.6f}  sentence=\"{best_step_sentence}\"")

    return " ".join(words), current_emb, current_dist


def _generate_random_sentence(vocab, length=10):
    return " ".join(random.sample(vocab, min(length, len(vocab))))


def find_distant_pair(model, seed_sentence, vocab,
                      max_outer_iters=1, max_inner_steps=50,
                      top_k_positions=3, top_k_words=50,
                      num_random_starts=1, verbose=True):
    sent_a = seed_sentence
    emb_a = embed_sentence(model, sent_a)
    sent_b = seed_sentence
    emb_b = embed_sentence(model, sent_b)
    best_dist = 0.0
    seed_len = len(seed_sentence.split())

    for sweep in range(max_outer_iters):
        if sweep % 2 == 0:
            anchor_emb = emb_a
            moving_sent = sent_b
        else:
            anchor_emb = emb_b
            moving_sent = sent_a

        starting_sentences = [moving_sent]
        if num_random_starts > 1:
            for _ in range(num_random_starts - 1):
                starting_sentences.append(_generate_random_sentence(vocab, seed_len))

        sweep_best_sent, sweep_best_emb, sweep_best_dist = None, None, -1.0

        for start_sent in starting_sentences:
            new_sent, new_emb, dist = greedy_word_replace(
                model, anchor_emb, start_sent, vocab,
                max_inner_steps=max_inner_steps,
                top_k_positions=top_k_positions,
                top_k_words=top_k_words,
                verbose=verbose,
            )
            if dist > sweep_best_dist:
                sweep_best_dist = dist
                sweep_best_sent = new_sent
                sweep_best_emb = new_emb

        if verbose:
            print(f"  Sweep {sweep}: pair distance = {sweep_best_dist:.6f}")

        if sweep % 2 == 0:
            sent_b, emb_b = sweep_best_sent, sweep_best_emb
        else:
            sent_a, emb_a = sweep_best_sent, sweep_best_emb

        if sweep_best_dist > best_dist + 1e-7:
            best_dist = sweep_best_dist
        else:
            if verbose:
                print(f"Sweep {sweep}: converged at {best_dist:.6f}")
            break

    return sent_a, sent_b, best_dist


# ==============================================================================
# Multi-restart wrapper
# ==============================================================================
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


def multi_restart_search(model, vocab, seed_sentences=None,
                         max_outer_iters=10, max_inner_steps=50,
                         top_k_positions=3, top_k_words=50,
                         num_random_starts=1, top_n=10, verbose=True):
    if seed_sentences is None:
        seed_sentences = SEED_SENTENCES

    top_heap = []
    for i, seed in enumerate(seed_sentences):
        print(f"\n{'#'*70}")
        print(f"# Restart {i+1}/{len(seed_sentences)}: \"{seed[:60]}...\"")
        print(f"{'#'*70}")
        s1, s2, dist = find_distant_pair(
            model, seed, vocab,
            max_outer_iters=max_outer_iters,
            max_inner_steps=max_inner_steps,
            top_k_positions=top_k_positions,
            top_k_words=top_k_words,
            num_random_starts=num_random_starts,
            verbose=verbose,
        )
        print(f"\n  Restart {i+1} best: dist={dist:.6f}")
        print(f"    S1: \"{s1}\"")
        print(f"    S2: \"{s2}\"")

        if len(top_heap) < top_n:
            heapq.heappush(top_heap, (dist, s1, s2))
        elif dist > top_heap[0][0]:
            heapq.heapreplace(top_heap, (dist, s1, s2))

    return sorted(top_heap, key=lambda x: -x[0])


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MSGS: Find maximally distant sentence pairs in embedding space.")
    parser.add_argument("--embedding_model", type=str)
    parser.add_argument("--vocab_source", type=str, default="nltk")
    parser.add_argument("--max_vocab", type=int, default=20000)
    parser.add_argument("--max_outer_iters", type=int, default=1)
    parser.add_argument("--max_inner_steps", type=int, default=50)
    parser.add_argument("--top_k_positions", type=int, default=3)
    parser.add_argument("--top_k_words", type=int, default=50)
    parser.add_argument("--num_random_starts", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="distant_pairs.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_sentence", type=str, default=None)
    parser.add_argument("--no_multi_restart", action="store_true")
    parser.add_argument("--random_seed_sentences", type=int, default=0)
    parser.add_argument("--seed_sentence_length", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    print("Loading vocabulary...")
    vocab = load_vocabulary(args.vocab_source, args.max_vocab)
    print(f"  Vocabulary size: {len(vocab)}")

    t0 = time.time()

    if args.random_seed_sentences > 0:
        seeds = [_generate_random_sentence(vocab, args.seed_sentence_length)
                 for _ in range(args.random_seed_sentences)]
    elif args.seed_sentence:
        seeds = [args.seed_sentence]
    else:
        seeds = None

    if seeds and len(seeds) == 1 and args.no_multi_restart:
        s1, s2, dist = find_distant_pair(
            model, seeds[0], vocab,
            max_outer_iters=args.max_outer_iters,
            max_inner_steps=args.max_inner_steps,
            top_k_positions=args.top_k_positions,
            top_k_words=args.top_k_words,
            num_random_starts=args.num_random_starts,
            verbose=args.verbose,
        )
        top_pairs = [(dist, s1, s2)]
    else:
        top_pairs = multi_restart_search(
            model, vocab,
            seed_sentences=seeds,
            max_outer_iters=args.max_outer_iters,
            max_inner_steps=args.max_inner_steps,
            top_k_positions=args.top_k_positions,
            top_k_words=args.top_k_words,
            num_random_starts=args.num_random_starts,
            top_n=args.top_n,
            verbose=args.verbose,
        )

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"TOP-{len(top_pairs)} DISTANT PAIRS (elapsed {elapsed:.1f}s)")
    print(f"{'='*70}")
    results = []
    for rank, (dist, s1, s2) in enumerate(top_pairs, 1):
        e1, e2 = embed_batch(model, [s1, s2])
        l2_dist = float(np.linalg.norm(e1 - e2))
        cos_sim = float(np.dot(e1, e2))
        print(f"\n  Rank {rank}:")
        print(f"    Euclidean distance: {l2_dist:.6f}")
        print(f"    Cosine similarity:  {cos_sim:.6f}")
        print(f"    S1: \"{s1}\"")
        print(f"    S2: \"{s2}\"")
        results.append({
            "rank": rank,
            "euclidean_distance": l2_dist,
            "cosine_similarity": cos_sim,
            "sentence_1": s1,
            "sentence_2": s2,
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
