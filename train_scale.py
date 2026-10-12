"""
Scale training script for ByteMinGramPPTokenizer on WikiText-2.
Downloads the corpus, mines byte-level seed candidates, and trains
the tokenizer with iterative pruning to a target vocabulary size.
"""

import urllib.request
import collections
import time
import sys
from mingram_pp import ByteMinGramPPTokenizer

CORPUS_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
CORPUS_PATH = "/tmp/wikitext2_valid.txt"
TARGET_VOCAB = 8000
MAX_SUBSTR_LEN = 12
PRUNE_FACTOR = 0.2


def download_corpus(path: str) -> str:
    print(f"Downloading corpus from {CORPUS_URL} ...")
    urllib.request.urlretrieve(CORPUS_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus loaded: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")
    return text


def mine_byte_candidates(lines: list[str], max_len: int = MAX_SUBSTR_LEN) -> dict[bytes, int]:
    """Extract byte-level substring candidates from the corpus lines."""
    counts: dict[bytes, int] = collections.Counter()
    total = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        b = stripped.encode("utf-8")
        for start in range(len(b)):
            for end in range(start + 2, min(start + max_len + 1, len(b) + 1)):
                counts[b[start:end]] += 1
        if (i + 1) % 1000 == 0:
            print(f"  Mining candidates: {i+1}/{total} lines ...")
    return counts


def run_inference_profile(tokenizer, test_lines: list[str], n: int = 100):
    """Profile compression ratio on held-out lines."""
    total_bytes = 0
    total_tokens = 0
    for line in test_lines[:n]:
        stripped = line.strip()
        if not stripped:
            continue
        encoded = tokenizer.encode(stripped)
        total_bytes += len(stripped.encode("utf-8"))
        total_tokens += len(encoded)
    if total_tokens > 0:
        print(f"  Bytes/Token: {total_bytes / total_tokens:.2f}")
        print(f"  Tokens/Byte: {total_tokens / total_bytes:.4f}")
    return total_bytes, total_tokens


def main():
    text = download_corpus(CORPUS_PATH)
    lines = text.splitlines()

    print(f"\nMining byte substrings (max_len={MAX_SUBSTR_LEN}) ...")
    t0 = time.time()
    candidate_counts = mine_byte_candidates(lines)
    print(f"Mining done in {time.time() - t0:.1f}s — {len(candidate_counts):,} unique candidates")

    print(f"\nLoading seed vocabulary ...")
    tokenizer = ByteMinGramPPTokenizer()
    tokenizer.load_seed_vocabulary(candidate_counts)
    print(f"Seed vocab size: {len(tokenizer.vocab):,}")

    print(f"\nTraining with iterative pruning to {TARGET_VOCAB} tokens ...")
    t0 = time.time()
    tokenizer.train_prune_bytes(
        lines,
        target_vocab_size=TARGET_VOCAB,
        prune_factor=PRUNE_FACTOR,
        em_steps_per_iter=2,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    print(f"\n--- INFERENCE PROFILE (WikiText-2 valid set) ---")
    run_inference_profile(tokenizer, lines, n=len(lines))

    print(f"\n--- SAMPLE TOKENIZATIONS ---")
    samples = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require efficient tokenizers",
        "Unigram language models use probabilistic segmentation",
        "Subword regularization improves neural machine translation",
    ]
    for s in samples:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        match = s == decoded
        print(f"  '{s}'")
        print(f"    tokens={len(tokens)}, ids={len(ids)}, roundtrip={match}")
        print(f"    path: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    print(f"\nFinal vocab size: {len(tokenizer.vocab):,}")


if __name__ == "__main__":
    main()