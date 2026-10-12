"""
Benchmark: ByteMinGramPPTokenizer vs SentencePiece Unigram.
Compares compression ratio, training time, and tokenization speed
on the same WikiText-2 corpus at equal vocabulary sizes.
"""

import time
import urllib.request
import sentencepiece as spm
import os
import sys

from mingram_pp import ByteMinGramPPTokenizer

CORPUS_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
CORPUS_PATH = "/tmp/wikitext2_valid.txt"
SP_MODEL_PREFIX = "/tmp/sp_unigram"
TARGET_VOCAB = 8000


def download_corpus(path: str) -> str:
    if not os.path.exists(path):
        print(f"Downloading corpus ...")
        urllib.request.urlretrieve(CORPUS_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} chars, {len(text.encode('utf-8')):,} bytes")
    return text


# ── SentencePiece ──────────────────────────────────────────────────

def train_sentencepiece(corpus_path: str, vocab_size: int) -> spm.SentencePieceProcessor:
    print(f"\n{'='*60}")
    print(f"Training SentencePiece Unigram (vocab={vocab_size})")
    print(f"{'='*60}")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=SP_MODEL_PREFIX,
        model_type="unigram",
        vocab_size=vocab_size,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{SP_MODEL_PREFIX}.model")
    print(f"Actual vocab size: {sp.get_piece_size()}")
    return sp


def profile_sp(sp, lines: list[str]) -> dict:
    total_bytes = 0
    total_tokens = 0
    t0 = time.time()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ids = sp.encode(stripped, out_type=int)
        total_bytes += len(stripped.encode("utf-8"))
        total_tokens += len(ids)
    elapsed = time.time() - t0
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(1, total_tokens),
        "tok_per_sec": total_tokens / max(0.001, elapsed),
        "time": elapsed,
    }


# ── MinGram-PP ─────────────────────────────────────────────────────

def train_mingram(lines: list[str], vocab_size: int) -> ByteMinGramPPTokenizer:
    import collections
    print(f"\n{'='*60}")
    print(f"Training ByteMinGramPPTokenizer (vocab={vocab_size})")
    print(f"{'='*60}")

    print("Mining byte substrings ...")
    counts: dict[bytes, int] = collections.Counter()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        b = stripped.encode("utf-8")
        for start in range(len(b)):
            for end in range(start + 2, min(start + 13, len(b) + 1)):
                counts[b[start:end]] += 1
    print(f"  {len(counts):,} unique candidates")

    tok = ByteMinGramPPTokenizer()
    tok.load_seed_vocabulary(counts)
    t0 = time.time()
    tok.train_prune_bytes(lines, target_vocab_size=vocab_size, prune_factor=0.2, em_steps_per_iter=2)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")
    return tok


def profile_mingram(tok, lines: list[str]) -> dict:
    total_bytes = 0
    total_tokens = 0
    t0 = time.time()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ids = tok.encode(stripped)
        total_bytes += len(stripped.encode("utf-8"))
        total_tokens += len(ids)
    elapsed = time.time() - t0
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(1, total_tokens),
        "tok_per_sec": total_tokens / max(0.001, elapsed),
        "time": elapsed,
    }


# ── Samples ────────────────────────────────────────────────────────

def show_samples(sp, mingram, samples: list[str]):
    print(f"\n{'='*60}")
    print("SAMPLE TOKENIZATIONS")
    print(f"{'='*60}")
    for s in samples:
        sp_ids = sp.encode(s, out_type=int)
        sp_tokens = [sp.id_to_piece(i) for i in sp_ids]
        mg_ids = mingram.encode(s)
        mg_tokens = mingram.tokenize(s)
        sp_decoded = sp.decode(sp_ids)
        mg_decoded = mingram.decode(mg_ids)

        print(f"\n'{s}'")
        print(f"  SentencePiece : {len(sp_tokens)} tokens | {sp_tokens[:12]}{'...' if len(sp_tokens)>12 else ''}")
        print(f"  MinGram-PP    : {len(mg_tokens)} tokens | {mg_tokens[:12]}{'...' if len(mg_tokens)>12 else ''}")
        print(f"  SP roundtrip  : {sp_decoded == s}")
        print(f"  MG roundtrip  : {mg_decoded == s}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    text = download_corpus(CORPUS_PATH)
    lines = text.splitlines()

    sp = train_sentencepiece(CORPUS_PATH, TARGET_VOCAB)
    mingram = train_mingram(lines, TARGET_VOCAB)

    sp_stats = profile_sp(sp, lines)
    mg_stats = profile_mingram(mingram, lines)

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS  (vocab={TARGET_VOCAB})")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'SentencePiece':>15} {'MinGram-PP':>15}")
    print(f"{'-'*55}")
    print(f"{'Bytes/Token':<25} {sp_stats['bytes_per_token']:>15.2f} {mg_stats['bytes_per_token']:>15.2f}")
    print(f"{'Tokens/Byte':<25} {1/sp_stats['bytes_per_token']:>15.4f} {1/mg_stats['bytes_per_token']:>15.4f}")
    print(f"{'Total Tokens':<25} {sp_stats['tokens']:>15,} {mg_stats['tokens']:>15,}")
    print(f"{'Tokenize Time (s)':<25} {sp_stats['time']:>15.1f} {mg_stats['time']:>15.1f}")
    print(f"{'Tokens/sec':<25} {sp_stats['tok_per_sec']:>15,.0f} {mg_stats['tok_per_sec']:>15,.0f}")

    samples = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require efficient tokenizers",
        "Unigram language models use probabilistic segmentation",
        "Subword regularization improves neural machine translation",
        "ArticlePubMedGoogle Scholar",
        "Héllo wörld café résumé naïve",
    ]
    show_samples(sp, mingram, samples)


if __name__ == "__main__":
    main()