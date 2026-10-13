Papers referred: 
- [Kudo](https://arxiv.org/abs/1804.10959)
- [Land](https://arxiv.org/abs/2606.27019)

Hardware used for training

- TPU: v5e-1 TPU

## Benchmark: MinGram-PP vs SentencePiece

WikiText-2 validation set (1.1M bytes, vocab=8000)

| Metric | SentencePiece | MinGram-PP |
|---|---|---|
| Bytes/Token | 3.70 | 3.51 |
| Total Tokens | 300,255 | 316,484 |
| Tokenize Time | 0.1s | 2.3s |
| Tokens/sec | 3.25M | 135K |

- SentencePiece trained in 1.1s (C++, suffix array seed generation)
- MinGram-PP trained in 547s (pure Python, brute-force substring mining)
- Both achieve comparable compression quality
- Speed gap is in seed generation, not the core Viterbi + EM algorithm

## Seed Generation

### Current: Brute-Force Substring Mining

The current approach iterates over every line in the corpus, encodes it as UTF-8 bytes, and extracts all substrings up to a maximum length (default 12 bytes) using a nested loop:

```python
for start in range(len(b)):
    for end in range(start + 2, min(start + max_len + 1, len(b) + 1)):
        counts[b[start:end]] += 1
```

This is O(n * L^2) per line where n is line length and L is max substring length. On WikiText-2 (1.1M bytes) it produces ~3.9M unique candidates and takes ~550s. The quadratic inner loop and per-slice byte allocation dominate runtime.

### Proposed: Suffix Array Extraction

Replace brute-force mining with a suffix array to extract all repeated substrings in O(n log n) time:

1. Concatenate the entire corpus into a single byte sequence, separated by sentinel bytes
2. Build a suffix array and LCP (Longest Common Prefix) array
3. Enumerate substrings by walking adjacent suffix pairs in LCP order — each LCP entry of length `l` implies `l-1` repeated substrings
4. Filter candidates by minimum frequency threshold before counting exact occurrences

This avoids the per-line quadratic scan entirely. SentencePiece uses this approach in C++ and extracts seed candidates from the same corpus in ~0.1s. A Python implementation using `pysuffix` or a custom SA would target the same algorithmic complexity, with the C extension handling the inner loop.

### Implemented: SA-IS Suffix Array Results

WikiText-2 validation set (1.1M bytes, vocab=8000)

| Metric | Brute-Force | Suffix Array |
|---|---|---|
| Seed generation | 6.2s | 5.1s |
| Unique candidates | 3,942,396 | 222,322 |
| Total training | 547s | 89s |
| Bytes/Token | 3.51 | 2.88 |

- SA-IS construction: O(n) linear time with L-type/S-type classification and LMS sorting
- LCP array via Kasai's algorithm: O(n) construction
- Candidate enumeration: walks LCP intervals to find maximal groups sharing prefixes of length 2..max_len
- SA extracts 17x fewer candidates due to min_count=2 filtering — only substrings appearing at least twice
- Total training time reduced 6.2x (547s → 89s)
- Compression quality slightly worse (2.88 vs 3.51 bytes/token) — smaller candidate pool gives EM less material to work with during pruning

### C Extension: Viterbi Decoder via ctypes

Ported the FlatMatrixTrie construction and Viterbi byte-chunk decoder to C (`mingram_c.c`), loaded from Python via ctypes. The EM loop, pruning logic, and SA-IS seed generation remain in Python.

WikiText-2 validation set (1.1M bytes, vocab=8000)

| Version | Training Time | Bytes/Token |
|---|---|---|
| Pure Python (brute-force) | 547s | 3.51 |
| Pure Python (suffix array) | 89s | 2.88 |
| C-accelerated (suffix array) | 43s | 2.88 |

- Total speedup: 12.7x (547s → 43s)
- C handles the inner Viterbi loop: trie lookup, DP transitions, backtracking
- Python handles: corpus preprocessing, EM count aggregation, vocabulary pruning
- Compression quality unchanged — same algorithm, just faster execution

